"""
CSV Logger Module
Logs recognition events to CSV files for analysis and reporting.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class RecognitionEvent:
    """Data class for recognition events."""
    timestamp: str
    frame_number: int
    face_id: Optional[int]
    name: str
    similarity: float
    bounding_box: List[int]  # [x, y, width, height]
    confidence: float
    source: str  # 'video', 'image', 'webcam', etc.
    source_path: Optional[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV serialization."""
        data = asdict(self)
        # Convert bounding box list to string
        data['bounding_box'] = json.dumps(data['bounding_box'])
        # Convert metadata to JSON string if present
        if data['metadata']:
            data['metadata'] = json.dumps(data['metadata'])
        return data


class CSVLogger:
    """CSV logger for face recognition events."""
    
    def __init__(self, 
                 log_dir: Optional[Path] = None,
                 max_file_size_mb: int = 10,
                 rotate_files: bool = True):
        """
        Initialize CSV logger.
        
        Args:
            log_dir: Directory to store log files (default: data/logs)
            max_file_size_mb: Maximum file size in MB before rotation
            rotate_files: Whether to rotate log files when they get too large
        """
        from .config import Config
        self.config = Config()
        
        self.log_dir = log_dir or (self.config.DATA_DIR / 'logs')
        self.max_file_size_mb = max_file_size_mb
        self.rotate_files = rotate_files
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_file: Optional[Path] = None
        self.writer: Optional[csv.DictWriter] = None
        self.file_handle = None
        
        # Field names for CSV
        self.fieldnames = [
            'timestamp', 'frame_number', 'face_id', 'name', 'similarity',
            'bounding_box', 'confidence', 'source', 'source_path', 'metadata'
        ]
        
        logger.info(f"CSV logger initialized. Log directory: {self.log_dir}")
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path, creating new one if needed."""
        if self.current_file is None or not self.current_file.exists():
            # Create new log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_file = self.log_dir / f"recognition_log_{timestamp}.csv"
            
            # Create file with header
            with open(self.current_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            
            logger.info(f"Created new log file: {self.current_file}")
        
        # Check if file needs rotation
        if self.rotate_files and self.current_file.exists():
            file_size_mb = self.current_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                # Rotate to new file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.current_file = self.log_dir / f"recognition_log_{timestamp}.csv"
                
                with open(self.current_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                
                logger.info(f"Rotated to new log file: {self.current_file}")
        
        return self.current_file
    
    def log_event(self, event: RecognitionEvent) -> bool:
        """
        Log a recognition event to CSV.
        
        Args:
            event: RecognitionEvent to log
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_file = self._get_current_log_file()
            
            # Open file in append mode
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(event.to_dict())
            
            logger.debug(f"Logged event: {event.name} (similarity: {event.similarity:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False
    
    def log_recognition_results(self, 
                               frame_number: int,
                               results: Dict,
                               source: str = 'video',
                               source_path: Optional[str] = None,
                               metadata: Optional[Dict] = None) -> int:
        """
        Log recognition results from pipeline.
        
        Args:
            frame_number: Frame number in video/stream
            results: Results from pipeline.process_frame()
            source: Source type ('video', 'image', 'webcam', etc.)
            source_path: Path to source file
            metadata: Additional metadata
            
        Returns:
            Number of events logged
        """
        events_logged = 0
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Log recognized faces
        if results.get('recognition_results'):
            for rec in results['recognition_results']:
                event = RecognitionEvent(
                    timestamp=timestamp,
                    frame_number=frame_number,
                    face_id=rec.get('face_id'),
                    name=rec.get('name', 'Unknown'),
                    similarity=rec.get('similarity', 0.0),
                    bounding_box=rec.get('box', [0, 0, 0, 0]),
                    confidence=rec.get('confidence', 0.0),
                    source=source,
                    source_path=source_path,
                    metadata=metadata
                )
                
                if self.log_event(event):
                    events_logged += 1
        
        # Log unknown faces (detected but not recognized)
        if results.get('detections'):
            for det in results['detections']:
                # Check if this detection was already recognized
                box = det.get('box')
                if box:
                    already_logged = False
                    if results.get('recognition_results'):
                        for rec in results['recognition_results']:
                            rec_box = rec.get('box')
                            if rec_box and rec_box == box:
                                already_logged = True
                                break
                    
                    if not already_logged:
                        event = RecognitionEvent(
                            timestamp=timestamp,
                            frame_number=frame_number,
                            face_id=None,
                            name='Unknown',
                            similarity=0.0,
                            bounding_box=box,
                            confidence=det.get('confidence', 0.0),
                            source=source,
                            source_path=source_path,
                            metadata=metadata
                        )
                        
                        if self.log_event(event):
                            events_logged += 1
        
        if events_logged > 0:
            logger.debug(f"Logged {events_logged} recognition events for frame {frame_number}")
        
        return events_logged
    
    def get_recent_events(self, 
                         limit: int = 100,
                         name_filter: Optional[str] = None,
                         min_similarity: float = 0.0) -> List[Dict]:
        """
        Get recent recognition events.
        
        Args:
            limit: Maximum number of events to return
            name_filter: Filter by name (case-insensitive)
            min_similarity: Minimum similarity score
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        try:
            # Get all log files sorted by modification time (newest first)
            log_files = sorted(self.log_dir.glob('recognition_log_*.csv'), 
                             key=lambda f: f.stat().st_mtime, 
                             reverse=True)
            
            for log_file in log_files:
                if len(events) >= limit:
                    break
                
                with open(log_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    
                    # Read rows in reverse order (newest first in file)
                    rows = list(reader)
                    for row in reversed(rows):
                        if len(events) >= limit:
                            break
                        
                        # Parse similarity
                        similarity = float(row.get('similarity', 0))
                        
                        # Apply filters
                        if similarity < min_similarity:
                            continue
                        
                        if name_filter:
                            name = row.get('name', '').lower()
                            if name_filter.lower() not in name:
                                continue
                        
                        # Parse bounding box from JSON
                        bbox_str = row.get('bounding_box', '[]')
                        try:
                            row['bounding_box'] = json.loads(bbox_str)
                        except:
                            row['bounding_box'] = []
                        
                        # Parse metadata from JSON if present
                        metadata_str = row.get('metadata', '')
                        if metadata_str:
                            try:
                                row['metadata'] = json.loads(metadata_str)
                            except:
                                row['metadata'] = {}
                        
                        events.append(row)
        
        except Exception as e:
            logger.error(f"Failed to read recent events: {e}")
        
        return events
    
    def get_statistics(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get statistics from log files.
        
        Args:
            start_time: Start time for statistics
            end_time: End time for statistics
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_events': 0,
            'known_faces': 0,
            'unknown_faces': 0,
            'avg_similarity': 0.0,
            'face_counts': {},
            'time_period': {}
        }
        
        total_similarity = 0.0
        
        try:
            log_files = list(self.log_dir.glob('recognition_log_*.csv'))
            
            for log_file in log_files:
                with open(log_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Parse timestamp
                        timestamp_str = row.get('timestamp', '')
                        try:
                            event_time = datetime.fromisoformat(timestamp_str)
                        except:
                            continue
                        
                        # Apply time filter
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue
                        
                        # Update statistics
                        stats['total_events'] += 1
                        
                        name = row.get('name', 'Unknown')
                        similarity = float(row.get('similarity', 0))
                        
                        if name.lower() == 'unknown':
                            stats['unknown_faces'] += 1
                        else:
                            stats['known_faces'] += 1
                            total_similarity += similarity
                        
                        # Count by name
                        if name not in stats['face_counts']:
                            stats['face_counts'][name] = 0
                        stats['face_counts'][name] += 1
            
            # Calculate average similarity (only for known faces)
            if stats['known_faces'] > 0:
                stats['avg_similarity'] = total_similarity / stats['known_faces']
            
            # Add time period info
            if start_time:
                stats['time_period']['start'] = start_time.isoformat()
            if end_time:
                stats['time_period']['end'] = end_time.isoformat()
        
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up logger resources."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
        
        self.file_handle = None
        self.writer = None
        self.current_file = None
        
        logger.info("CSV logger cleaned up")