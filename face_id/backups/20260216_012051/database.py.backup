"""
Face Database Module
SQLite database for storing face embeddings and metadata.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Database for managing face embeddings and metadata."""
    
    def __init__(self, db_path: str = 'data/faces.db'):
        """
        Initialize the database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_db_directory()
    
    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> None:
        """Connect to the database."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def initialize(self) -> None:
        """Create database tables."""
        self.connect()
        
        cursor = self.conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP,
                seen_count INTEGER DEFAULT 1
            )
        ''')
        
        # Create index on name for faster searches
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)
        ''')
        
        # Create face_images table (for storing multiple images per person)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER NOT NULL,
                image_path TEXT,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
            )
        ''')
        
        # Create detection_history table (for tracking detections)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                confidence REAL,
                quality REAL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE SET NULL
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def add_face(self, name: str, embedding: np.ndarray, 
                 metadata: Optional[Dict] = None) -> int:
        """
        Add a new face to the database.
        
        Args:
            name: Person name
            embedding: Face embedding vector (numpy array)
            metadata: Additional metadata (will be stored as JSON)
            
        Returns:
            Face ID
        """
        self.connect()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        embedding_dim = len(embedding)
        
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO faces (name, embedding, embedding_dim, metadata)
            VALUES (?, ?, ?, ?)
        ''', (name, embedding_bytes, embedding_dim, metadata_json))
        
        face_id = cursor.lastrowid
        self.conn.commit()
        
        logger.info(f"Added face: {name} (ID: {face_id})")
        return face_id
    
    def get_face(self, face_id: int) -> Optional[Dict]:
        """
        Get face by ID.
        
        Args:
            face_id: Face ID
            
        Returns:
            Face data dict or None if not found
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
        
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_dict(row)
    
    def get_face_by_name(self, name: str) -> Optional[Dict]:
        """
        Get face by name.
        
        Args:
            name: Person name
            
        Returns:
            Face data dict or None if not found
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM faces WHERE name = ?', (name,))
        
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_dict(row)
    
    def find_matching_face(self, embedding: np.ndarray, 
                          threshold: float = 0.5) -> Optional[Tuple[Dict, float]]:
        """
        Find matching face by embedding using cosine similarity.
        
        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold (0-1)
            
        Returns:
            Tuple of (face_data, similarity) or None if no match
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM faces')
        
        best_match = None
        best_similarity = 0.0
        
        for row in cursor.fetchall():
            face_data = self._row_to_dict(row)
            stored_embedding = face_data['embedding']
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = face_data
        
        if best_match:
            logger.debug(f"Found match: {best_match['name']} (similarity: {best_similarity:.3f})")
            return best_match, best_similarity
        
        return None
    
    def list_faces(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """
        List all faces in the database.
        
        Args:
            limit: Maximum number of faces to return
            offset: Number of faces to skip
            
        Returns:
            List of face data dicts
        """
        self.connect()
        
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute('SELECT * FROM faces ORDER BY created_at DESC LIMIT ? OFFSET ?', 
                          (limit, offset))
        else:
            cursor.execute('SELECT * FROM faces ORDER BY created_at DESC')
        
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def update_face(self, face_id: int, name: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Update face information.
        
        Args:
            face_id: Face ID
            name: New name (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False otherwise
        """
        self.connect()
        
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if name is not None:
            updates.append('name = ?')
            params.append(name)
        
        if metadata is not None:
            updates.append('metadata = ?')
            params.append(json.dumps(metadata))
        
        if not updates:
            return False
        
        updates.append('updated_at = ?')
        params.append(datetime.now().isoformat())
        
        params.append(face_id)
        
        query = f"UPDATE faces SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def update_last_seen(self, face_id: int) -> None:
        """
        Update last seen timestamp and increment seen count.
        
        Args:
            face_id: Face ID
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE faces 
            SET last_seen_at = ?, 
                seen_count = seen_count + 1,
                updated_at = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), datetime.now().isoformat(), face_id))
        
        self.conn.commit()
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from the database.
        
        Args:
            face_id: Face ID
            
        Returns:
            True if deleted, False otherwise
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        
        self.conn.commit()
        
        deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted face ID: {face_id}")
        
        return deleted
    
    def get_face_count(self) -> int:
        """Get total number of faces in database."""
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM faces')
        
        return cursor.fetchone()[0]
    
    def add_detection_history(self, face_id: Optional[int], confidence: float,
                             quality: float, image_path: Optional[str] = None) -> int:
        """
        Add a detection to history.
        
        Args:
            face_id: Face ID (None for unknown faces)
            confidence: Detection confidence
            quality: Face quality score
            image_path: Path to saved image (optional)
            
        Returns:
            Detection history ID
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO detection_history (face_id, confidence, quality, image_path)
            VALUES (?, ?, ?, ?)
        ''', (face_id, confidence, quality, image_path))
        
        self.conn.commit()
        
        return cursor.lastrowid
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert database row to dictionary with embedding as numpy array."""
        data = dict(row)
        
        # Convert embedding bytes to numpy array
        if data['embedding']:
            data['embedding'] = np.frombuffer(data['embedding'], dtype=np.float32)
        
        # Parse metadata JSON
        if data['metadata']:
            data['metadata'] = json.loads(data['metadata'])
        
        return data
    
    @staticmethod
    def _cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        return f"FaceDatabase(db_path='{self.db_path}')"
