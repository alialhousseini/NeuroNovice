from ResNet.Resnet import ResNet
import torch

import torch.onnx as onnx

def generate_onnx(model, output_file):
    """
    Generates an ONNX file of the visualized PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be visualized.
        output_file (str): The path to save the generated ONNX file.

    Returns:
        None
    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    input_names = ['Image']
    output_names = ['Predicted Label']

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, output_file,
                      input_names=input_names, output_names=output_names, verbose=True)


mod = ResNet()
generate_onnx(mod, "ResNet/resnet.onnx")
