import time
from argparse import ArgumentParser
import cv2

from search_images import ImageSearcher
from optimize_utils import OrderSolver
from app_config import Config


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Required. Path to a image.",
                        required=True,
                        type=str)

    return parser.parse_args()


def main():
    args = build_argparser()
    conf_file = "config_double.yml"
    conf = Config(conf_file)

    image_searcher = ImageSearcher(
        conf.model_obj_path, conf.model_scene_path,
        conf.sqlite_path, conf.ngt_obj_path, conf.ngt_scene_path)

    image = cv2.imread(args.input)
    start = time.time()
    f_pathes, dists, similarity_matrix = image_searcher.search_related(
        image, 10)
    process_time = time.time() - start
    print("search time", process_time, " sec")
    results = []
    for f_path, dist in zip(f_pathes, dists):
        print(f_path, dist)
        results.append([f_path, dist])

    rebalancer = OrderSolver()
    start = time.time()
    success, results = rebalancer.rebalance_order(
        similarity_matrix, results, omega=2, M=4)
    process_time = time.time() - start
    print("order optimizing time", process_time, " sec")
    print("success", success)
    for f_path, dist in results:
        print(f_path, dist)


if __name__ == '__main__':
    main()
