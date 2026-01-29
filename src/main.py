import cv2
import yaml

from video.reader import VideoReader
from inference.pytorch_backend import PyTorchBackend


def draw_detections(frame, detections):
    for x1, y1, x2, y2, score, class_id in detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_id}:{score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

    while True:
        frame, fps = reader.read()
        if frame is None:
            break

        detections = backend.infer(frame)
        frame = draw_detections(frame, detections)

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
