class InferenceBackend:
    def infer(self, frame):
        """
        Args:
            frame (np.ndarray): BGR image
        Returns:
            detections (list): [x1, y1, x2, y2, score, class_id]
        """
        raise NotImplementedError
