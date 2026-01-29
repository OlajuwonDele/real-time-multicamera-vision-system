from ultralytics import YOLO


class PyTorchBackend:
    def __init__(self, model_name, device="cpu"):
        self.model = YOLO(model_name)
        self.model.to(device)

    def infer(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, score, class_id])

        return detections
