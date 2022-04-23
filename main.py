import os

import config as conf
import utils

if __name__ == '__main__':
    MODEL = utils.load_torch_model(conf.MODEL, conf.INPUT_SIZE)
    if conf.MODEL.upper() == 'ALL':
        for key in MODEL.keys():
            print('exporting: {}'.format(key))
            utils.export_onnx(
                MODEL[key], os.path.join(conf.WEIGHT_PATH, key + '.onnx')
            )
    elif MODEL is not None:
        print('exporting: {}'.format(key))
        utils.export_onnx(
            MODEL, os.path.join(conf.WEIGHT_PATH, conf.MODEL.upper())
        )
