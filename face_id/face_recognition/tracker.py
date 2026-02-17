"""
Face Tracker Module
Simple IoU-based face tracking between frames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Track:
    """Represents a single tracked face."""
    
    def __init__(self, track_id: int, bbox: List[int], confidence: float):
        """
        Initialize a track.
        
        Args:
            track_id: Unique track ID
            bbox: Bounding box [x, y, width, height]
            confidence: Detection confidence
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0  # Frames since last update
        self.total_hits = 1
        self.hit_streak = 1
    
    def update(self, bbox: List[int], confidence: float) -> None:
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            confidence: New confidence
        """
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0
        self.total_hits += 1
        self.hit_streak += 1
    
    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.age += 1
        self.hit_streak = 0
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (has enough hits)."""
        return self.hit_streak >= 3
    
    def is_dead(self, max_age: int) -> bool:
        """Check if track should be removed."""
        return self.age > max_age


class FaceTracker:
    """Face tracker for maintaining face IDs across frames."""
    
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3,
                 min_hits: int = 3):
        """
        Initialize the face tracker.
        
        Args:
            max_age: Maximum frames to keep a track alive without updates
            iou_threshold: IoU threshold for track association
            min_hits: Minimum hits to confirm a track
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of face detections, each with 'box' and 'confidence'
            
        Returns:
            List of tracks with IDs, each with 'track_id', 'box', 'confidence', 'age', 'hits'
        """
        self.frame_count += 1
        
        # If no detections, age all tracks
        if len(detections) == 0:
            tracks_to_remove = []
            
            for track_id, track in self.tracks.items():
                track.mark_missed()
                
                if track.is_dead(self.max_age):
                    tracks_to_remove.append(track_id)
            
            # Remove dead tracks
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                logger.debug(f"Removed dead track {track_id}")
            
            return self._get_active_tracks()
        
        # If no existing tracks, create new tracks for all detections
        if len(self.tracks) == 0:
            for detection in detections:
                self._create_new_track(detection)
            
            return self._get_active_tracks()
        
        # Match detections to existing tracks using IoU
        matched, unmatched_detections, unmatched_tracks = self._associate(detections)
        
        # Update matched tracks
        for track_id, detection_idx in matched:
            detection = detections[detection_idx]
            self.tracks[track_id].update(
                detection['box'],
                detection.get('confidence', 1.0)
            )
        
        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            self._create_new_track(detection)
        
        # Remove dead tracks
        tracks_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.is_dead(self.max_age)
        ]
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            logger.debug(f"Removed dead track {track_id}")
        
        return self._get_active_tracks()
    
    def _associate(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], 
                                                           List[int], 
                                                           List[int]]:
        """
        Associate detections with existing tracks using IoU.
        
        Args:
            detections: List of detections
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id].bbox
            
            for j, detection in enumerate(detections):
                detection_bbox = detection['box']
                iou_matrix[i, j] = self._calculate_iou(track_bbox, detection_bbox)
        
        # Greedy matching
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Find best match
            best_iou = 0
            best_track_idx = -1
            best_detection_idx = -1
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_track_idx = i
                        best_detection_idx = j
            
            # If best IoU exceeds threshold, create match
            if best_iou >= self.iou_threshold:
                matched.append((track_ids[best_track_idx], best_detection_idx))
                unmatched_tracks.remove(best_track_idx)
                unmatched_detections.remove(best_detection_idx)
            else:
                # No more valid matches
                break
        
        # Convert unmatched track indices to track IDs
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matched, unmatched_detections, unmatched_track_ids
    
    def _create_new_track(self, detection: Dict) -> int:
        """
        Create a new track from a detection.
        
        Args:
            detection: Detection dict with 'box' and 'confidence'
            
        Returns:
            New track ID
        """
        track_id = self.next_id
        self.next_id += 1
        
        track = Track(
            track_id=track_id,
            bbox=detection['box'],
            confidence=detection.get('confidence', 1.0)
        )
        
        self.tracks[track_id] = track
        logger.debug(f"Created new track {track_id}")
        
        return track_id
    
    def _get_active_tracks(self) -> List[Dict]:
        """
        Get list of active tracks.
        
        Returns:
            List of track dicts
        """
        active_tracks = []
        
        for track_id, track in self.tracks.items():
            # Only return confirmed tracks or tracks with recent hits
            if track.is_confirmed() or track.hit_streak > 0:
                active_tracks.append({
                    'track_id': track.track_id,
                    'box': track.bbox,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.total_hits,
                    'hit_streak': track.hit_streak,
                    'confirmed': track.is_confirmed()
                })
        
        return active_tracks
    
    @staticmethod
    def _calculate_iou(box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            box1: [x, y, width, height]
            box2: [x, y, width, height]
            
        Returns:
            IoU score (0-1)
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all tracks."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    def __repr__(self) -> str:
        return (f"FaceTracker(max_age={self.max_age}, "
                f"iou_threshold={self.iou_threshold}, "
                f"active_tracks={len(self.tracks)})")
