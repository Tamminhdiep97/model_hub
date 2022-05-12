import os
import time

import numpy as np
from tqdm import tqdm

import config as conf
import utils


def test_time(model):
    sum_time = 0
    for _ in tqdm(range(0, conf.NUM_SAMPLES)):
        input_tensor = np.random.randn(1, 3, 112, 112).astype(np.float32)
        time_start = time.time()
        _ = model.run(
            None,
            {'input_1': input_tensor},
        )
        time_end = time.time()
        sum_time += time_end - time_start
    print('TEST ON {} SAMPLES DONE'.format(conf.NUM_SAMPLES))
    print(
        'Each [1, 3, 112, 112] numpy tensor took: {}s on avarage'.format(
            sum_time / conf.NUM_SAMPLES
        )
    )


if __name__ == '__main__':
    for _file in os.listdir(conf.WEIGHT_PATH):
        if _file.find('onnx') != -1:
            path_model = os.path.join(conf.WEIGHT_PATH, _file)
            print('Load model: {}'.format(_file))
            model = utils.load_model(path_model)
            test_time(model)
