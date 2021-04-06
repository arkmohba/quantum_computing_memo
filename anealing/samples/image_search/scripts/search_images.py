
import numpy as np
import logging as log
from openvino.inference_engine import IECore

import sqlalchemy
from sqlalchemy.orm import sessionmaker
import ngtpy
import scipy

from add_vector2ngt import prepare_net, prepare_images, infer
from model.image_feature import ImageDoubleFeature


class ModelHelper:
    def __init__(self, ie, model_xml, batch_size, device):
        self.net, self.input_blob, self.out_blob = prepare_net(
            ie, model_xml, batch_size)
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape
        self.output_shape = self.net.outputs[self.out_blob].shape
        self.exec_net = ie.load_network(network=self.net, device_name=device)


class ImageSearcher:
    def __init__(self, model_xml_obj, model_xml_scene,
                 sqlite_path, ngt_path_obj, ngt_path_scene):
        self.engine = sqlalchemy.create_engine(sqlite_path)
        self.index_obj = ngtpy.Index(ngt_path_obj)
        self.index_scene = ngtpy.Index(ngt_path_scene)

        # OpenVINOの設定
        ie = IECore()
        batch_size = 1
        device = "CPU"
        self.model_obj = ModelHelper(ie, model_xml_obj, batch_size, device)
        self.model_scene = ModelHelper(ie, model_xml_scene, batch_size, device)

    def search_related(self, img, n_search,
                       search_for="objective", arrange_for="scenary"):
        self.session = sessionmaker(bind=self.engine)()
        if search_for == "objective":
            use_model = self.model_obj
            use_index = self.index_obj
            target_feat_id = ImageDoubleFeature.feat_objective_id
        else:
            use_model = self.model_scene
            use_index = self.index_scene
            target_feat_id = ImageDoubleFeature.feat_scene_id

        # 推論を実行
        images = prepare_images(img, use_model.input_shape)
        result = infer(images, use_model.exec_net,
                       use_model.input_blob, use_model.out_blob)

        # NGTから近い画像を検索する
        query = result[0]
        feat_objs = use_index.search(query, n_search)
        feat_ids = [obj[0] for obj in feat_objs]
        feat_dists = [obj[1] for obj in feat_objs]

        # sqlから画像パスと特徴ベクトルを取り出す
        results = self.session.query(ImageDoubleFeature).filter(
            target_feat_id.in_(feat_ids))
        if search_for == "objective":
            # 検索先のIDをキーにする
            oid_2_entity = {res.feat_objective_id: res for res in results}
        else:
            oid_2_entity = {res.feat_scene_id: res for res in results}
        # スコアで整列されたファイルパス
        f_pathes = [oid_2_entity[f_id].f_path for f_id in feat_ids]

        # 並び替えに使用する特徴ベクトル
        if arrange_for == "objective":
            feat_vectors = [np.frombuffer(
                oid_2_entity[f_id].feat_obj, dtype=np.float32)
                for f_id in feat_ids]
        else:
            feat_vectors = [np.frombuffer(
                oid_2_entity[f_id].feat_scene, dtype=np.float32)
                for f_id in feat_ids]
        # 候補どうしの類似度行列
        dist_matrix = np.zeros((n_search, n_search))
        for i in range(n_search - 1):
            for j in range(i + 1, n_search):
                if arrange_for == "objective":
                    # コサイン距離
                    dist_matrix[i][j] = scipy.spatial.distance.cosine(
                        feat_vectors[i], feat_vectors[j])
                else:
                    # scene向けであればL2ノルム
                    dist_matrix[i][j] = np.linalg.norm(
                        feat_vectors[i] - feat_vectors[j])
        # 距離から類似度に変換
        similarity_matrix = np.exp(- dist_matrix / dist_matrix.max())

        return f_pathes, feat_dists, np.array(similarity_matrix)
