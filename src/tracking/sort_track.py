from libs.sort import Sort
from tracking.base_tracker import Tracker
import numpy as np

class SortTracker(Tracker):
    def __init__(self, class_mapping):
        """
        Args:
            class_mapping (dict): {class_id: class_name} from the YOLO model
        """
        self.class_mapping = class_mapping
        # Create a tracker for every class name found in the model
        self.trackers = {
            name: Sort() 
            for name in class_mapping.values()}

    def track(self):
        if self.frame is None or self.detections is None:
            return []
        
        all_tracks = []
        
        for name, tracker in self.trackers.items():
            # Filter detections where class_name (index 6) matches the tracker's class
            class_dets = [d[:5] for d in self.detections if d[6] == name]
            
            # Update specific tracker (even if empty to maintain track history)
            track_input = np.array(class_dets, dtype=np.float32) if class_dets else np.empty((0,5), dtype=np.float32)

            tracks = tracker.update(track_input)    
            for t in tracks:
                x1, y1, x2, y2, track_id = t

                all_tracks.append({
                    "id": int(track_id),
                    "ltrb": [float(x1), float(y1), float(x2), float(y2)],
                    "class_name": name,
                    "is_confirmed": True  # SORT has no confirmation state
                })
        
        return all_tracks