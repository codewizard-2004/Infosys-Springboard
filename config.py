import os
from dotenv import load_dotenv
import torch

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    APP_NAME = "FoodVision: Food Classification using PyTorch and Groq"
    GROQ_API_KEY = os.getenv("PUBLIC_GROQ_API_KEY")
    TORCH_PATH = os.path.join(ROOT_DIR, "models", "torch")
    ONNX_PATH = {
        "lenet64": os.path.join(ROOT_DIR, "models", "onnx", "lenet64.onnx"),
        "tinyvgg": os.path.join(ROOT_DIR, "models", "onnx", "tinyvgg.onnx"),
        "resnet18": os.path.join(ROOT_DIR, "models", "onnx", "resnet18.onnx")
    }
    NUTRIENTS_PATH = os.path.join(ROOT_DIR, "data", "nutrients.csv")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ARCH_PATH = {
        "lenet64": os.path.join(ROOT_DIR, "architecture", "lenet64.py"),
        "tinyvgg": os.path.join(ROOT_DIR, "architecture", "tinyvgg.py"),
    }
    CLASS_NAMES = ["pizza", "steak", "sushi"]


settings = Config()