from deep_sort_realtime.deepsort_tracker import DeepSort
from tracking.base_tracker import Tracker

class DeepsortTracker(Tracker):
    def __init__(self):
        self.object_tracker = DeepSort()

    def track(self):
        if self.frame is None or self.detections is None:
            return []
        
        formatted_detections = []
        class_names = []
        for x1, y1, x2, y2, score, class_id, class_name in self.detections:
            width = x2 - x1
            height = y2 - y1
            
            class_names.append(class_name)
            
            bbox = [float(x1), float(y1), float(width), float(height)]
            formatted_detections.append((bbox, float(score), int(class_id)))

        tracks = self.object_tracker.update_tracks(formatted_detections, frame=self.frame)
        formatted_tracks = []
   
        for track, name in zip(tracks, class_names):
            if not track.is_confirmed():
                continue
            
            formatted_tracks.append({
                "id": track.track_id,
                "ltrb": track.to_ltrb(), # Returns [x1, y1, x2, y2]
                "is_confirmed": True,
                "class_name": name,
            })

        return formatted_tracks
  