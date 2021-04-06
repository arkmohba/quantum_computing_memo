import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                        type=str)
    parser.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                        required=True,
                        type=str)
    parser.add_argument("-d", "--device",
                        help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                        "acceptable. The sample will look for a suitable plugin for device specified. Default "
                        "value is CPU",
                        default="CPU", type=str)

    return parser.parse_args()


def main():
    args = build_argparser()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    net.batch_size = 10

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        print(args.input)
        image = cv2.imread(args.input)
        if image.shape[:-1] != (h, w):
            log.warning(
                f"Image {args.input} is resized from {image.shape[:-1]} to {(h, w)}")
            image = cv2.resize(image, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        images[i] = image

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    infer_request_handle = exec_net.start_async(
        request_id=0, inputs={input_blob: image})
    infer_status = infer_request_handle.wait()
    res = infer_request_handle.output_blobs[out_blob]
    print(res.buffer.shape)


if __name__ == "__main__":
    main()
