import cv2
import yaml

from video.reader import VideoReader
from inference.pytorch_backend import PyTorchBackend
from tracking.deepsort import DeepsortTracker

def draw_detections(deepsort_tracker, frame, detections):
    deepsort_tracker.update(frame, detections)
    tracks, class_names = deepsort_tracker.track()
    for track, class_name in zip(tracks, class_names):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        label = f" ID: {str(track_id)}, {class_name}" 
        cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(frame, label, (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    return frame

def main():
    with open("src/config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    reader = VideoReader(
        source=config["video"]["source"],
        width=config["video"]["width"],
        height=config["video"]["height"]
    )

    backend = PyTorchBackend(
        model_name=config["model"]["name"],
        device=config["model"]["device"]
    )

    deepsort_tracker = DeepsortTracker()
    while True:
        frame, fps = reader.read()
        if frame is None:
            break

        detections = backend.infer(frame)
        frame = draw_detections(deepsort_tracker, frame, detections)
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
