import onnxruntime as ort
import torch
from torchsummary import summary

from backbone import ghost_model, ir_ghost_model, ir_model


def export_onnx(model, onnx_path):
    summary(model, (3, 112, 112))
    dummy_input = torch.randn(1, 3, 112, 112, device=torch.device('cpu'))
    input_names = ['input_1']
    output_names = ['output_1']
    dynamic_axes = {
        'input_1': [0, 2, 3],
        'output_1': {0: 'output_1_variable_dim_0', 1: 'output_1_variable_dim_1'},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return None


def load_model(path):
    model = ort.InferenceSession(path)
    return model


def load_torch_model(model_name, input_size):
    irse_model = ir_model.model_irse
    resnet_model = ir_model.model_resnet
    irse_model_ghost = ir_ghost_model.model_ir_ghost
    model_dict = {
        'IR_50': irse_model.ir_50(input_size),
        'IR_101': irse_model.ir_101(input_size),
        'IR_152': irse_model.ir_152(input_size),
        'IRSE_50': irse_model.ir_se_50(input_size),
        'IRSE_101': irse_model.ir_se_101(input_size),
        'IRSE_152': irse_model.ir_se_152(input_size),
        'IR_GHOST_50': irse_model_ghost.ir_ghost_50(input_size),
        'IR_GHOST_101': irse_model_ghost.ir_ghost_101(input_size),
        'IR_GHOST_152': irse_model_ghost.ir_ghost_152(input_size),
        'IRSE_GHOST_50': irse_model_ghost.irse_ghost_50(input_size),
        'IRSE_GHOST_101': irse_model_ghost.irse_ghost_101(input_size),
        'IRSE_GHOST_152': irse_model_ghost.irse_ghost_152(input_size),
        'RESNET_50': resnet_model.resnet_50(input_size),
        'RESNET_101': resnet_model.resnet_101(input_size),
        'RESNET_152': resnet_model.resnet_152(input_size),
        'GHOSTNET': ghost_model.model_ghostnet.GhostNet(),
    }

    if model_name.upper() in model_dict.keys():
        return model_dict[model_name.upper()]
    elif model_name.upper() == 'ALL':
        return model_dict
    else:
        return None
