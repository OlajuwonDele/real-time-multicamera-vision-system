import time
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np

from video.reader import VideoReader
from typing import Optional

class InferencePerformance:
    def __init__(
        self,
        pytorch_model: str,
        onnx_model: Optional[str] = None,
        tensorrt_model: Optional[str] = None,
        data_yaml: Optional[str] = None,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cuda",
    ):
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA requested but not available"

        self.device = device
        
        self.pytorch_path = pytorch_model
        self.pytorch_model = YOLO(pytorch_model).to(device) if pytorch_model else None

        self.onnx_path = onnx_model
        onnx_file = Path(self.onnx_path)
        self.onnx_model = YOLO(onnx_model, task='detect') if onnx_model and onnx_file.exists() else None
        
        self.tensorrt_path = tensorrt_model
        engine_file = Path(self.tensorrt_path)
        self.tensorrt_model = YOLO(tensorrt_model, task='detect') if tensorrt_model and engine_file.exists() else None

        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.batch = batch

        self.create_all_models(onnx_file, engine_file)

    def create_all_models(self, onnx_file, engine_file):
        if self.pytorch_model is None:
            raise ValueError("PyTorch model (.pt) is required as base")

        if onnx_file and not onnx_file.exists():
            print(f"ONNX file {self.onnx_path} not found. Exporting...")
            self.pytorch_model.export(format="onnx")
        
        if onnx_file and self.onnx_model is None:
            print(f"Loading ONNX model from {self.onnx_path}")
            self.onnx_model = YOLO(self.onnx_path, task='detect')
        
        if engine_file and not engine_file.exists():
            print(f"TensorRT file {self.tensorrt_path} not found. Exporting...")
            self.pytorch_model.export(format="engine", device=self.device)
        
        if engine_file and self.tensorrt_model is None:
            print(f"Loading TensorRT model from {self.tensorrt_path}")
            self.tensorrt_model = YOLO(self.tensorrt_path, task='detect')
        
        print("Model verification complete.")

    def model_run(self, model_path, frame):
        ext = Path(model_path).suffix

        if ext == ".pt":
            return self.pytorch_model(frame, verbose=False)[0]

        elif ext == ".onnx":
            return self.onnx_model(frame, verbose=False)[0]

        elif ext == ".engine":
            return self.tensorrt_model(frame, verbose=False)[0]

        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def evaluate_accuracy(self, model_path):
        # We use the path stored in the config, not the reader
        if not self.data_yaml:
            print("Skipping Accuracy: No data_yaml provided in config.")
            return {"mAP50": 0, "mAP50-95": 0, "precision": 0, "recall": 0}

        model = YOLO(model_path, task='detect')

        # model.val() handles loading the dataset from self.data_yaml
        metrics = model.val(
            data=self.data_yaml,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            verbose=False,
            plots=False # Speeds up validation
        )

        return {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.precision),
            "recall": float(metrics.box.recall),
        }


    def evaluate_speed(
        self,
        model_path,
        reader: VideoReader,
        warmup_frames: int = 10,
        n_frames: int = 200,
    ):
        times = []

        # Warmup 
        for _ in range(warmup_frames):
            frame, _ = reader.read()
            if frame is not None:
                self.model_run(model_path, frame)
            
        for _ in range(n_frames):
            frame, _ = reader.read()
            
            if frame is None: 
                break 

            t0 = time.time()
            self.model_run(model_path, frame)
            times.append(time.time() - t0)

        if not times: return {"latency_ms": 0, "fps": 0}

        mean_latency = sum(times) / len(times)

        return {
            "latency_ms": mean_latency * 1000,
            "fps": 1.0 / mean_latency,
        }
    
    def evaluate_all(self, reader):
        results = {}

        models = {
            "PyTorch": self.pytorch_path,
            "ONNX": self.onnx_path,
            "TensorRT": self.tensorrt_path
        }

        for name, model_path in models.items():
            if model_path is None:
                continue

            print(f"\nEvaluating {name.upper()} model: {model_path}")

            acc = self.evaluate_accuracy(model_path)
            speed = self.evaluate_speed(model_path, reader)

            results[name] = {
                "model": model_path,
                **acc,
                **speed,
            }

        return results

