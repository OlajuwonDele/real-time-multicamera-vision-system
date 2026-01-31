from tracking.base_tracker import Tracker
import yolox 
from yolox.tracker.byte_tracker import BYTETracker, STrack
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


np.float = float 
class ByteTracker(Tracker):
    def __init__(self,class_mapping):
        """
        Args:
            class_mapping (dict): {class_id: class_name} from the YOLO model
        """
        self.object_tracker = BYTETracker(BYTETrackerArgs())
        self.class_mapping = class_mapping
        # Create a tracker for every class name found in the model
        self.trackers = {
            name: BYTETracker(BYTETrackerArgs()) 
            for name in class_mapping.values()}
        
    def track(self):
        if self.frame is None or self.detections is None:
            return []
        
        all_tracks = []
        img_info = self.frame.shape[:2]
        
        for name, tracker in self.trackers.items():
            # Filter detections where class_name (index 6) matches the tracker's class
            class_dets = [d[:5] for d in self.detections if d[6] == name]
            
            # Update specific tracker (even if empty to maintain track history)
            track_input = np.array(class_dets) if len(class_dets) > 0 else np.empty((0, 5))
            class_targets = tracker.update(track_input, img_info, img_info)

            for track in class_targets:
                all_tracks.append({
                    "id": track.track_id,
                    "ltrb": track.tlbr,
                    "class_name": name,
                    "is_confirmed": track.is_activated
                })
        
        return all_tracks