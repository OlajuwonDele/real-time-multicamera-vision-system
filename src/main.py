import cv2
import yaml
        
import time
import numpy as np
from typing import Dict, List

from video.reader import VideoReader

from inference.pytorch_backend import PyTorchBackend
from inference.onnx_backend import ONNXBackend
from inference.tensorrt_backend import TensorRTBackend
from inference.inference_performance import InferencePerformance

from tracking.deepsort import DeepsortTracker
from tracking.bytetrack import ByteTracker
from tracking.sort_track import SortTracker


def best_backend(config = None, performance_reader = None):
    results = model_performance_benchmark(performance_test=True, reader=performance_reader, config=config)

    best_model_name = min(results, key=lambda k: results[k]["latency_ms"])
    
    print(f"Best backend selected: {best_model_name} ({results[best_model_name]['latency_ms']:.2f} ms)")

    if best_model_name == "PyTorch":
        return PyTorchBackend(
            model_name=config["custom_dataset_pytorch_model"]["name"],
            device=config["pytorch_model"]["device"]
        )
        
    elif best_model_name == "TensorRT":
        return TensorRTBackend(
            model_name=config["custom_dataset_tensorrt_model"]["name"],
            device=config["custom_dataset_tensorrt_model"]["device"]
        )

    elif best_model_name == "ONNX":
        return ONNXBackend(
            model_name=config["custom_dataset_onnx_model"]["name"],
            device=config["custom_dataset_onnx_model"]["device"]
        )
    
    else:
        raise ValueError(f"Unknown model type selected: {best_model_name}")


def benchmark_trackers(
    trackers: Dict[str, object], 
    reader, 
    backend, 
    n_frames: int = 50
) -> Dict[str, float]:

    
    # Pre-compute detections to isolate tracking performance
    data_cache = []
    
    for _ in range(n_frames):
        frame, _ = reader.read()
        if frame is None:
            break
        detections = backend.infer(frame)
        data_cache.append((frame, detections))

    if not data_cache:
        raise ValueError("VideoReader returned no frames for benchmarking.")

    results = {}
    
    for name, tracker in trackers.items():
        latencies = []
        
        for frame, detections in data_cache:
            tracker.update(frame, detections)
            
            t0 = time.time()
            _ = tracker.track()
            t1 = time.time()
            
            latencies.append((t1 - t0) * 1000) # Convert to ms

        avg_latency = sum(latencies) / len(latencies)
        results[name] = avg_latency
        print(f"Tracker: {name:<15} | Latency: {avg_latency:.2f} ms | FPS: {1000/avg_latency:.1f}")

    return results

def best_tracker(model_classes: dict, reader, detection_model) -> object:
    """
    Instantiates available trackers, benchmarks them, and returns the fastest one.
    """
    trackers = {
        "ByteTrack": ByteTracker(class_mapping=model_classes),
        "SORT": SortTracker(class_mapping=model_classes),
        "DeepSORT": DeepsortTracker() 
    }

    # 2. Run Benchmark
    # We pass the reader and model to generate test data
    scores = benchmark_trackers(trackers, reader, detection_model)

    # 3. Select Winner
    # We select the one with the minimum latency
    fastest = min(scores, key=scores.get)
    fastest_tracker = trackers[fastest]

    print(f"\n Best Tracker: {fastest_tracker}")
    
    return fastest_tracker

def draw_detections(tracker, frame, detections):
    tracker.update(frame, detections)
    tracks = tracker.track()
    for track in tracks:
        if not track['is_confirmed']:
            continue
        ltrb = track['ltrb']
        bbox = ltrb
        label = f"ID: {track['id']}, {track['class_name']}" 
        cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(frame, label, (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    return frame

def model_performance_benchmark(performance_test = False, reader = None, config = None):
    if performance_test == True:
            infer_performance = InferencePerformance(
                pytorch_model=config["custom_dataset_pytorch_model"]["name"], 
                onnx_model=config["custom_dataset_onnx_model"]["name"],
                tensorrt_model=config["custom_dataset_tensorrt_model"]["name"], 
                device=config["custom_dataset_pytorch_model"]["device"],
                data_yaml=config["dataset"]["data_yaml"],
                )
            return infer_performance.evaluate_all(reader)
    return {}
     



     
def main():
    with open("src/config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    reader = VideoReader(
        source=config["video"]["source"],
        width=config["video"]["width"],
        height=config["video"]["height"]
    )

    # backend.train(dataset_yaml=config["dataset"]["data_yaml"])  # train on custom dataset. In this project I use a basketball game.
    performance_reader = reader
    backend = best_backend(config = config, performance_reader = performance_reader)
    model_classes = backend.model.names
    tracker = best_tracker(model_classes, performance_reader, backend)
    
    
    while True:
        frame, fps = reader.read()
        if frame is None:
            break

        detections = backend.infer(frame)
        frame = draw_detections(tracker, frame, detections)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if config["runtime"]["display"]:
            cv2.imshow("Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
