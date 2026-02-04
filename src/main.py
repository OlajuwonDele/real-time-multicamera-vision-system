import cv2
import yaml

from video.reader import VideoReader

from inference.pytorch_backend import PyTorchBackend
from inference.onnx_backend import ONNXBackend
from inference.tensorrt_backend import TensorRTBackend
from inference.inference_performance import InferencePerformance

from tracking.deepsort import DeepsortTracker
from tracking.bytetrack import ByteTracker
from tracking.sort_track import SortTracker


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

def performance_benchmark(performance_test = False, reader = None, config = None):
    if performance_test == True:
            infer_performance = InferencePerformance(
                pytorch_model=config["pytorch_model"]["name"], 
                onnx_model=config["onnx_model"]["name"],
                tensorrt_model=config["tensorrt_model"]["name"], 
                device=config["pytorch_model"]["device"],
                data_yaml=config["dataset"]["data_yaml"],
                )
            infer_performance.evaluate_all(reader)
     
def main():
    with open("src/config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    reader = VideoReader(
        source=config["video"]["source"],
        width=config["video"]["width"],
        height=config["video"]["height"]
    )

    performance_reader = reader
    performance_test = False
    performance_benchmark(performance_test, performance_reader, config)

    backend = PyTorchBackend(
        model_name=config["pytorch_model"]["name"],
        device=config["pytorch_model"]["device"]
    )
    
    # backend = TensorRTBackend(
    #     model_name=config["tensorrt_model"]["name"],
    #     device=config["tensorrt_model"]["device"]
    # )

    # backend = ONNXBackend(
    #     model_name=config["pytorch_model"]["name"],
    #     device=config["pytorch_model"]["device"]
    # )
    model_classes = backend.model.names
    print(model_classes)
    # backend.train(dataset_yaml=config["dataset"]["data_yaml"])
    # model_classes = backend.model.names
    # print(model_classes)
    # tracker = DeepsortTracker()
    # tracker = ByteTracker(class_mapping=model_classes)
    tracker = SortTracker(class_mapping=model_classes)
     
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
