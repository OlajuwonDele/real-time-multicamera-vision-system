from ultralytics import YOLO
from inference.inference_backend import InferenceBackend

class PyTorchBackend(InferenceBackend):
    def __init__(self, model_name, device="cuda"):
        self.model = YOLO(model_name)
        self.model.to(device)

    def infer(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for result in results:
            class_names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                detections.append([x1, y1, x2, y2, score, class_id, class_name])

        return detections
    
    @property
    def names(self):
        return self.model.names
    
    def train(self, dataset_yaml):
         results = self.model.train(
        data=dataset_yaml,          # Path to data config
        epochs=100,               
        imgsz=640,                  
        batch=16,                   
        device=0,                   # Use 0 for your first CUDA GPU
        project='runs/train',       # Where to save results
        name='basketball_model',    # Name of this training run
        optimizer='auto'            # 'SGD', 'Adam', etc.
    )