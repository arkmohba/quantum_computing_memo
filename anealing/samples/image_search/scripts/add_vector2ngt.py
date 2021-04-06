import glob
import os
import shutil
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

import sqlalchemy
from sqlalchemy.orm import sessionmaker

import ngtpy
from model.image_feature import Base, ImageDoubleFeature
from tqdm import tqdm
from app_config import Config


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="Required. Path to a folder with images or path to an image files",
                        required=True,
                        type=str)

    return parser.parse_args()


def prepare_net(ie, model, batch_size_):
    model_bin = os.path.splitext(model)[0] + ".bin"

    net = ie.read_network(model=model, weights=model_bin)

    net.batch_size = batch_size_

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    return net, input_blob, out_blob


def prepare_images(image, input_shape):
    n, c, h, w = input_shape
    images = np.zeros(shape=(n, c, h, w), dtype=np.uint8)

    if image.shape[:-1] != (h, w):
        image = cv2.resize(image, (w, h))
    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    images[0] = image
    return images


def infer(input_data, exec_net, input_blob, out_blob, async_api=True):
    if async_api:
        infer_request_handle = exec_net.start_async(
            request_id=0, inputs={input_blob: input_data})
        infer_request_handle.wait()
        res = infer_request_handle.output_blobs[out_blob]
        return res.buffer
    else:
        res = exec_net.infer(inputs={input_blob: input_data})
        return res


class ModelHelper:
    def __init__(self, ie, model_xml, batch_size, device):
        self.net, self.input_blob, self.out_blob = prepare_net(
            ie, model_xml, batch_size)
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape
        self.output_shape = self.net.outputs[self.out_blob].shape
        self.exec_net = ie.load_network(network=self.net, device_name=device)


def main():
    args = build_argparser()
    conf_file = "config_double.yml"
    conf = Config(conf_file)

    # OpenVINOの設定
    ie = IECore()
    batch_size = 1
    exec_net_obj = ModelHelper(
        ie, conf.model_obj_path, batch_size, "CPU")
    exec_net_scene = ModelHelper(
        ie, conf.model_scene_path, batch_size, "CPU")

    # SQLAlchemyの設定
    if os.path.exists(conf.sqlite_path):
        os.remove(conf.sqlite_path)
    sqlite_path = conf.sqlite_path
    engine = sqlalchemy.create_engine(
        sqlite_path, echo=False)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()

    # NGTの設定
    # NGT1
    shutil.rmtree(conf.ngt_obj_path, ignore_errors=True)
    ngt_path = conf.ngt_obj_path
    ngtpy.create(
        ngt_path, exec_net_obj.output_shape[1], distance_type="Cosine")
    index_obj = ngtpy.Index(ngt_path)
    # NGT2
    shutil.rmtree(conf.ngt_scene_path, ignore_errors=True)
    ngt_path = conf.ngt_scene_path
    ngtpy.create(ngt_path, exec_net_scene.output_shape[1], distance_type="L2")
    index_scene = ngtpy.Index(ngt_path)

    # Start sync inference
    input_files = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    files = []
    results_obj = []
    results_scene = []
    feat_index_obj = []
    feat_index_scene = []
    for i, input_file in enumerate(tqdm(input_files)):
        files.append(input_file)
        # 推論を実行
        image = cv2.imread(input_file)
        # 物体ベースの特徴を取得
        images = prepare_images(image, exec_net_obj.input_shape)
        result = infer(images, exec_net_obj.exec_net,
                       exec_net_obj.input_blob, exec_net_obj.out_blob)
        results_obj.append(result[0])
        feat_index_obj.append(index_obj.insert(result[0]))  # ngtに登録＆ID発行
        # 模様ベースの特徴を取得
        images = prepare_images(image, exec_net_scene.input_shape)
        result = infer(images, exec_net_scene.exec_net,
                       exec_net_scene.input_blob, exec_net_scene.out_blob)
        results_scene.append(result[0])
        feat_index_scene.append(index_scene.insert(result[0]))  # ngtに登録＆ID発行

        if i > 0 and i % 5000 == 0:
            # バッチをまとめてsqliteに諸々登録
            session.add_all(instances=[
                ImageDoubleFeature(
                    f_path=f_path,
                    feat_objective_id=feat_id_obj,
                    feat_scene_id=feat_id_scene,
                    feat_obj=result_obj.tobytes(),
                    feat_scene=result_scene.tobytes())
                for f_path, feat_id_obj, feat_id_scene,
                result_obj, result_scene in zip(
                    files, feat_index_obj, feat_index_scene,
                    results_obj, results_scene)
            ])
            session.commit()
            files = []
            results_obj = []
            results_scene = []
            feat_index_obj = []
            feat_index_scene = []
    # あまりを登録
    # sqliteに諸々登録
    session.add_all(instances=[
        ImageDoubleFeature(
                    f_path=f_path,
                    feat_objective_id=feat_id_obj,
                    feat_scene_id=feat_id_scene,
                    feat_obj=result_obj.tobytes(),
                    feat_scene=result_scene.tobytes())
        for f_path, feat_id_obj, feat_id_scene,
        result_obj, result_scene in zip(
            files, feat_index_obj, feat_index_scene,
            results_obj, results_scene)
    ])
    session.commit()
    index_obj.build_index()
    index_obj.save()
    index_scene.build_index()
    index_scene.save()


if __name__ == "__main__":
    main()
