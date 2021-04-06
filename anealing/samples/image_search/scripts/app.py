import os
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import numpy as np
import cv2
from PIL import Image
from search_images import ImageSearcher
from app_config import Config
from optimize_utils import OrderSolver
import base64


APP_CONFIG = Config("config_double.yml")
app = Flask(__name__, static_folder=APP_CONFIG.static_dir)
bootstrap = Bootstrap(app)

image_searcher = ImageSearcher(
    APP_CONFIG.model_obj_path,
    APP_CONFIG.model_scene_path,
    APP_CONFIG.sqlite_path,
    APP_CONFIG.ngt_obj_path,
    APP_CONFIG.ngt_scene_path)

order_optimizer = OrderSolver()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_img_receive():
    # データからcsvファイルのデータを取り出して一時保存
    stream = request.files['image_file'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    # 取り出す候補の個数
    n_search = int(request.form['n_search'])
    score_for = request.form['score_for']
    sort_for = request.form['sort_for']

    # 関連するアイテムを取り出し
    f_pathes, dists, similarity_mat = image_searcher.search_related(
        img, n_search, search_for=score_for, arrange_for=sort_for)
    thum_pathes = [fpath.replace(APP_CONFIG.images_dir,
                                 APP_CONFIG.tumnail_dir) for fpath in f_pathes]
    for i, thum_path in enumerate(thum_pathes):
        if os.path.exists(thum_path):
            continue
        # 存在しない場合はサムネイルを生成
        image = Image.fromarray(cv2.cvtColor(
            cv2.imread(f_pathes[i]), cv2.COLOR_BGR2RGB))
        image.thumbnail((100, 100))
        image.save(thum_path, 'JPEG')
    results = list(zip(thum_pathes, dists))

    # 並び替え
    success, sorted_results = order_optimizer.rebalance_order(
        similarity_mat, results, M=5)
    results = [(a, b, c, d) for (a, b), (c, d) in zip(results, sorted_results)]
    # if success:
    #     results = tmp_results

    # クエリ画像をサムネイルにして返す
    query_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    query_img.thumbnail((100, 100))
    query_img = np.asarray(query_img)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
    result, query_img = cv2.imencode(
        '.jpg', query_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    query_img = base64.b64encode(query_img.tobytes()).decode("utf-8")
    query_img = "data:image/jpeg;base64,{}".format(query_img)

    return render_template('index.html', results=results, query_img=query_img)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
