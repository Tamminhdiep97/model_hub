import os
import time

import numpy as np
import onnxruntime as ort

import config as conf


def load_model(path):
    model = ort.InferenceSession(path)
    return model


def test_time(model):
    sum_time = 0
    for _ in range(0, conf.NUM_SAMPLES):
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
        'Time inference avarage for each [1, 3, 112, 112] numpy tensor: {}s'.format(
            sum_time / conf.NUM_SAMPLES
        )
    )


if __name__ == '__main__':
    for _file in os.listdir(conf.WEIGHT_PATH):
        if _file.find('onnx') != -1:
            path_model = os.path.join(conf.WEIGHT_PATH, _file)
            print('Load model: {}'.format(_file))
            model = load_model(path_model)
            test_time(model)
