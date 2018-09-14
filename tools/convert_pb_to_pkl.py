from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os
import pprint
import sys

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python import cnn

from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import detectron.utils.net as net_utils
import detectron.modeling.detector as detector

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = setup_logging(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a trained network to pb format')
    parser.add_argument(
        '--init_net_path',
        dest='init_net_path',
        help='Pretrained network weights file path',
        default=None,
        type=str)
    parser.add_argument(
        '--predict_net_path',
        dest='predict_net_path',
        help='Pretrained network weights file path',
        default=None,
        type=str)
    parser.add_argument(
        '--net_name',
        dest='net_name',
        help='optional name for the net',
        default="detectron",
        type=str)
    parser.add_argument(
        '--out_dir', dest='out_dir', help='output dir', default=None, type=str)
    parser.add_argument(
        '--net_execution_type',
        dest='net_execution_type',
        help='caffe2 net execution type',
        choices=['simple', 'dag'],
        default='simple',
        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    ret = parser.parse_args()
    ret.out_dir = os.path.abspath(ret.out_dir)

    return ret


def main():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    # with open(args.init_net_path) as f:
    # init_net = f.read()
    # with open(args.predict_net_path) as f:
    # predict_net = f.read()
    # p = workspace.Predictor(init_net, predict_net)
    # img = np.zeros((1,3,256,256), dtype=np.float32)
    # workspace.FeedBlob('data', img)
    # results = p.run({'data': img})

    init_def = caffe2_pb2.NetDef()
    with open(args.init_net_path, 'r') as f:
        init_def.ParseFromString(f.read())
        # init_def.device_option.CopyFrom(device_options)
    net_def = caffe2_pb2.NetDef()
    with open(args.predict_net_path, 'r') as f:
        net_def.ParseFromString(f.read())
        # net_def.device_option.CopyFrom(device_options)

    # model = model_helper.ModelHelper(arg_scope=arg_scope)
    # model = cnn.CNNModelHelper()
    model = detector.DetectionModelHelper(
        name=net_def.name, train=True, num_classes=1000, init_params=True)
    predict_net = core.Net(net_def)
    init_net = core.Net(init_def)
    model.param_init_net.AppendNet(init_net)
    model.net.AppendNet(predict_net)
    model.params.extend([
        core.BlobReference(x) for x in predict_net.Proto().external_input
        if x != 'data'
    ])

    # add_training_operators(model, 'pred', 'label')

    blob_names = ['data', 'label']
    for gpu_id in range(1):
        with c2_utils.NamedCudaScope(gpu_id):
            for blob_name in blob_names:
                workspace.CreateBlob(core.ScopedName(blob_name))

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    out_file_name = os.path.join(args.out_dir, net_def.name + '.pkl')
    net_utils.save_model_to_weights_file(out_file_name, model)

    # workspace.CreateNet(init_def)
    # workspace.CreateNet(net_def)
    # workspace.RunNet(net_def)
    # workspace.RunNet(init_def)

    print(type(init_def))
    print(net_def.name)

    print(workspace.blobs)
    print(len(workspace.blobs))
    print(workspace.Blobs())


if __name__ == '__main__':
    main()
