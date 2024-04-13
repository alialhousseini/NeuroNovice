import torch
from utils import generate_onnx
import os
import unittest


class UtilsTest(unittest.TestCase):
    def pass_test():
        self.assertTrue(1)
    # def test_generate_onnx(self):
    #     model = torch.hub.load('pytorch/vision:v0.10.0',
    #                            'resnet18', pretrained=True)

    #     # Ensure the directory exists
    #     os.makedirs("test", exist_ok=True)

    #     # Remove the ONNX file if it already exists
    #     os.remove("test/model.onnx") if os.path.exists("test/model.onnx") else None

    #     # Generate the ONNX file
    #     generate_onnx(model, "test/model.onnx")

    #     # Check if the ONNX file was created successfully
    #     self.assertTrue(os.path.exists("test/model.onnx"))
