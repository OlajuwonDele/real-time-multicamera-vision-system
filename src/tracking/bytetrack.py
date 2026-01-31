from tracking.tracker import Tracker
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
    def __init__(self):
        self.object_tracker = BYTETracker(BYTETrackerArgs())

    def track(self):
        if self.frame is None or self.detections is None:
            return []
        
        class_names = []
        data = []

        # self.detections: [x1, y1, x2, y2, score, class_id, class_name]
        for det in self.detections:
            data.append([det[0], det[1], det[2], det[3], det[4]])
            class_names.append(det[-1])
            
        track_input = np.array(data, dtype=float)
        tracks = self.object_tracker.update(output_results=track_input,
            img_info=self.frame.shape[:2],
            img_size=self.frame.shape[:2])

        formatted_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            formatted_tracks.append({
                "id": track.track_id,
                "ltrb": track.tlbr, # [x1, y1, x2, y2]
                "is_confirmed": True,
                "class_name": "name",
            })
            
        return formatted_tracks