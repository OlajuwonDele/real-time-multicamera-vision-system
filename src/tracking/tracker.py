class Tracker:
    def __init__(self):
        self.frame = None
        self.detections = None

    def update(self, frame, detections):
        self.frame = frame
        self.detections = detections
        
    
    def track(self):
        """
        Args:
            frame (np.ndarray): BGR image
            detections (list): [x1, y1, x2, y2, score, class_id, class_name]

            Returns: tracks, class_name
        """
        raise NotImplementedError