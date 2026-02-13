"""
Face Tracker Module
Simple IoU-based face tracking between frames.
"""

class FaceTracker:
    """Face tracker for maintaining face IDs across frames."""
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        """
        Initialize the face tracker.
        
        Args:
            max_age: Maximum frames to keep a track alive
            iou_threshold: IoU threshold for track association
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of face detections
            
        Returns:
            Updated tracks with IDs
        """
        # TODO: Implement tracking logic
        raise NotImplementedError("Tracker not yet implemented")
    
    def reset(self):
        """Reset all tracks."""
        self.tracks = {}
        self.next_id = 1
