import os

from ptflops import get_model_complexity_info

import config as conf
import utils

if __name__ == '__main__':
    MODEL = utils.load_torch_model(conf.MODEL, conf.INPUT_SIZE)
    if conf.MODEL.upper() == 'ALL':
        for key in MODEL.keys():
            macs, params = get_model_complexity_info(
                MODEL[key],
                (3, conf.INPUT_SIZE[0], conf.INPUT_SIZE[1]),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False,
            )

            print('Flop: {}, param: {}'.format(macs, params))
            # flops unit G, para unit M
            print('exporting: {}'.format(key))
            model_path = os.path.join(conf.WEIGHT_PATH, key + '.onnx')
            utils.export_onnx(MODEL[key], model_path)

    elif MODEL is not None:
        macs, params = get_model_complexity_info(
            MODEL,
            (3, conf.INPUT_SIZE[0], conf.INPUT_SIZE[1]),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )

        print('Flop: {}, param: {}'.format(macs, params))
        # flops unit G, para unit M
        print('exporting: {}'.format(conf.MODEL))
        model_path = os.path.join(conf.WEIGHT_PATH, conf.MODEL.upper() + '.onnx')
        utils.export_onnx(MODEL, model_path)
