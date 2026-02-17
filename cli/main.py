"""
CLI Main Module
Enhanced command-line interface for face recognition system.
"""

import sys
import os
import click
import cv2
import json
from pathlib import Path
from typing import Optional, List
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition import (
    RecognitionPipeline,
    VideoProcessor,
    Config
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Face Recognition CLI - Real-time face identification system with enhanced features.
    
    Available commands:
      start        Start real-time face detection and recognition
      add-face      Add a new face to the database
      list-faces    List all enrolled faces
      delete-face   Delete a face from the database
      process-video Process a video file for face recognition
      stats         Show database statistics
      export-faces  Export face database to JSON/CSV
      search        Search faces by name or ID
      batch-enroll  Batch enroll faces from a directory
      import-faces  Import faces from another database
      config        Manage configuration
      help          Show help information
    """
    pass


@click.command()
@click.option('--source', '-s', default='0', help='Video source: camera index (0-9), video file path, or RTSP/HTTP URL')
@click.option('--backend', '-b', type=click.Choice(['auto', 'v4l2', 'msmf', 'dshow', 'avfoundation', 'cv2']), default='auto', help='Video capture backend')
@click.option('--save-faces', type=click.Path(), help='Directory to save detected face images')
@click.option('--show-fps', is_flag=True, help='Show FPS counter')
@click.option('--list-cameras', is_flag=True, help='List available cameras and exit')
@click.option('--no-display', is_flag=True, help='Run without display (headless mode)')
@click.option('--max-frames', type=int, help='Maximum frames to process (for testing)')
def start(source, backend, save_faces, show_fps, list_cameras, no_display, max_frames):
    """
    Start real-time face detection and recognition.
    
    Args:
        source: Video source - camera index (0-9), video file path, or RTSP stream URL
        backend: Video capture backend (auto, v4l2, msmf, dshow, etc.)
        output: Optional output video file path
        no_display: Disable video display window
        save_faces: Directory to save detected face images
        show_fps: Whether to show FPS counter
        list_cameras: List available cameras
    """
    
    # List cameras if requested
    if list_cameras:
        list_available_cameras()
        return
    
    click.echo("="*60)
    click.echo("📹 FACE RECOGNITION - REAL-TIME MODE")
    click.echo("="*60)
    click.echo(f"Source: {source}")
    click.echo(f"Backend: {backend}")
    if save_faces:
        click.echo(f"Save faces: {save_faces}")
    if show_fps:
        click.echo(f"Show FPS counter: {show_fps}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Create video processor
        processor = VideoProcessor(pipeline=pipeline, config=config)
        
        # Determine source type and process accordingly
        source_type, source_path = parse_video_source(source)
        
        display = not no_display
        
        if source_type == 'camera':
            click.echo(f"Using camera index: {source_path}")
            processor.process_camera(
                camera_id=int(source_path),
                display=display,
                save_faces_dir=str(save_faces) if save_faces else None,
                show_fps=show_fps
            )
        elif source_type == 'video':
            click.echo(f"Processing video file: {source_path}")
            processor.process_video_file(
                input_path=str(source_path),
                display=display,
                max_frames=max_frames
            )
        elif source_type == 'rtsp':
            click.echo(f"Processing RTSP stream: {source_path}")
            processor.process_camera(
                camera_id=str(source_path),  # Pass URL as string
                display=display,
                is_rtsp=True,
                save_faces_dir=str(save_faces) if save_faces else None,
                show_fps=show_fps
            )
        
    except KeyboardInterrupt:
        click.echo("\n\n✓ Stopped by user")
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


def parse_video_source(source: str):
    """
    Parse video source string to determine type and path.
    
    Args:
        source: Source string (camera index, file path, or RTSP URL)
    
    Returns:
        Tuple of (source_type, source_path)
    """
    source = source.strip()
    
    # Check if it's an RTSP/HTTP stream URL
    if source.lower().startswith(('rtsp://', 'http://', 'https://')):
        return ('rtsp', source)
    
    # Check if it's a file path
    if os.path.isfile(source):
        return ('video', source)
    
    # Check if it's a camera index (digits only)
    if source.isdigit():
        return ('camera', int(source))
    
    # Try as camera index
    try:
        idx = int(source)
        return ('camera', idx)
    except ValueError:
        pass
    
    # Default to camera
    return ('camera', 0)


def list_available_cameras():
    """List available video capture devices."""
    click.echo("="*60)
    click.echo("📷 AVAILABLE CAMERAS")
    click.echo("="*60)
    
    import cv2
    
    # Try different backends
    backends = [
        (cv2.CAP_V4L2, 'V4L2'),
        (cv2.CAP_FFMPEG, 'FFMPEG'),
        (cv2.CAP_DSHOW, 'DirectShow'),
        (cv2.CAP_MSMF, 'MSMF'),
    ]
    
    found_cameras = []
    
    for backend_code, backend_name in backends:
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, backend_code)
                if cap.isOpened():
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    found_cameras.append({
                        'index': i,
                        'backend': backend_name,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                    click.echo(f"Camera {i} ({backend_name}): {width}x{height} @ {fps:.1f} FPS")
                cap.release()
            except Exception as e:
                pass
    
    # Try ffmpeg /dev/video devices directly
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-f', 'v4l2', '-list_devices', 'true', '-i', '/dev/null'], 
                              capture_output=True, text=True, timeout=5)
        if result.stderr:
            click.echo("\nFFmpeg V4L2 devices:")
            for line in result.stderr.split('\n'):
                if 'Video' in line or '/dev/video' in line:
                    click.echo(f"  {line.strip()}")
    except:
        pass
    
    if not found_cameras:
        click.echo("No cameras found. Try:")
        click.echo("  - Check if camera is physically connected")
        click.echo("  - Using video file: --source /path/to/video.mp4")
        click.echo("  - Using network/IP camera: --source rtsp://camera-ip:554/stream")
        click.echo("  - Using HTTP stream: --source http://camera-ip:8080/video")
    
    return found_cameras


@click.command()
@click.option('--name', required=True, help='Person name')
@click.option('--image', required=True, type=click.Path(exists=True), help='Image file path')
@click.option('--metadata', type=str, help='Additional metadata (JSON format)')
def add_face(name, image, metadata):
    """
    Add a new face to the database.
    
    Args:
        name: Person name
        image: Image file path
        metadata: Additional metadata in JSON format
    """
    click.echo("="*60)
    click.echo("👤 ADD KNOWN FACE")
    click.echo("="*60)
    click.echo(f"Name: {name}")
    click.echo(f"Image: {image}")
    if metadata:
        click.echo(f"Metadata: {metadata}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Store original image path
        image_path = image
        
        # First, detect faces in the image
        import cv2
        image_array = cv2.imread(image_path)
        if image_array is None:
            click.echo(f"\n❌ Error: Could not read image file")
            return
            
        results = pipeline.process_frame(image_array, recognize=False)
        detections = results.get('detections', [])
        
        click.echo(f"   Faces detected: {len(detections)}")
        
        if not detections:
            click.echo(f"\n⚠️  No face detected in the image!")
            return
        
        # Add face using original path
        face_id = pipeline.add_known_face_from_file(name, image_path, metadata=metadata)
        
        click.echo(f"\n✅ Face enrolled successfully!")
        click.echo(f"   Face ID: {face_id}")
        click.echo(f"   Database: {config.DB_PATH}")
        
    except ValueError as e:
        click.echo(f"\n⚠️  {str(e)}", err=True)
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
def list_faces():
    """List all enrolled faces."""
    click.echo("="*60)
    click.echo("👥 LIST ENROLLED FACES")
    click.echo("="*60)
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Get faces
        faces = pipeline.get_known_faces()
        
        if not faces:
            click.echo("\n⚠️  No faces in database")
            return
        
        # Display header
        click.echo(f"\n{'ID':<6} {'Name':<30} {'Created':<20} {'Seen':<10}")
        click.echo("-" * 80)
        
        # Display each face
        for face in faces:
            created = face.get('created_at', 'Never')[:19]
            updated = face.get('updated_at', 'Never')[:19]
            seen_count = face.get('seen_count', 0)
            last_seen = face.get('last_seen_at', 'Never')[:19] if face.get('last_seen_at') else 'Never'
            
            click.echo(
                f"{face['id']:<6} "
                f"{face['name']:<30} "
                f"{created:<20} "
                f"{updated:<20} "
                f"{seen_count:<10} "
                f"{last_seen:<19}"
            )
        
        # Display total
        click.echo("-" * 80)
        click.echo(f"\nTotal: {len(faces)} faces")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--face-id', required=True, type=int, help='Face ID to delete')
def delete_face(face_id):
    """
    Delete a face from the database.
    
    Args:
        face_id: Face ID to delete
    """
    click.echo("="*60)
    click.echo("🗑️ DELETE FACE")
    click.echo("="*60)
    click.echo(f"Face ID: {face_id}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Delete face
        deleted = pipeline.remove_known_face(face_id)
        
        if deleted:
            click.echo("✅ Face deleted successfully")
        else:
            click.echo("⚠️  Face ID not found")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--input', required=True, type=click.Path(), help='Input video file path')
@click.option('--output', required=True, type=click.Path(), help='Output video file path')
@click.option('--skip-frames', default=0, type=int, help='Number of frames to skip between processing')
@click.option('--no-display', is_flag=True, help='Disable video display window')
def process_video(input, output, skip_frames, no_display):
    """
    Process a video file for face recognition.
    
    Args:
        input: Input video file path
        output: Output video file path
        skip_frames: Number of frames to skip between processing
        no_display: Disable video display window
    """
    click.echo("="*60)
    click.echo("🎬️️ PROCESS VIDEO FILE")
    click.echo("="*60)
    click.echo(f"Input: {input}")
    click.echo(f"Output: {output}")
    click.echo(f"Skip frames: {skip_frames}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Create video processor
        processor = VideoProcessor(pipeline=pipeline, config=config)
        
        # Process video
        with click.progressbar(length=100, label='Processing frames') as bar:
            stats = processor.process_video_file(
                input_path=str(input),
                output_path=str(output),
                skip_frames=skip_frames,
                display=not no_display,
                callback=lambda frame, results: None
            )
            
        # Display statistics
        click.echo("")
        click.echo("="*60)
        click.echo("📊 PROCESSING STATISTICS")
        click.echo("="*60)
        click.echo(f"Total frames: {stats.get('total_frames', 0)}")
        click.echo(f"Processed frames: {stats.get('processed_frames', 0)}")
        click.echo(f"Elapsed time: {stats.get('elapsed_time', 0):.2f}s")
        click.echo(f"Average FPS: {stats.get('avg_fps', 0):.1f}")
        click.echo(f"Known faces detected: {stats.get('pipeline_stats', {}).get('known_faces', 0)}")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--output', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Output format')
def export_faces(output, format):
    """
    Export face database to file.
    
    Args:
        output: Output file path
        format: Output format (json or csv)
    """
    click.echo("="*60)
    click.echo("💾 EXPORT FACES")
    click.echo("="*60)
    click.echo(f"Output: {output}")
    click.echo(f"Format: {format.upper()}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Get faces
        faces = pipeline.get_known_faces()
        
        if not faces:
            click.echo("\n⚠️  No faces to export")
            return
        
        click.echo(f"Exporting {len(faces)} faces...")
        
        # Export based on format
        if format == 'json':
            # Create JSON output
            export_data = []
            for face in faces:
                export_data.append({
                    'id': face['id'],
                    'name': face['name'],
                    'created_at': face.get('created_at', ''),
                    'updated_at': face.get('updated_at', ''),
                    'seen_count': face.get('seen_count', 0),
                    'last_seen_at': face.get('last_seen_at', '') if face.get('last_seen_at') else ''
                })
            
            import json
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            click.echo(f"\n✅ Exported {len(export_data)} faces to JSON")
        
        elif format == 'csv':
            # Create CSV output
            import csv
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'Created At', 'Updated At', 'Seen Count', 'Last Seen At'])
                
                for face in faces:
                    writer.writerow([
                        face['id'],
                        face['name'],
                        face.get('created_at', ''),
                        face.get('updated_at', ''),
                        face.get('seen_count', 0),
                        face.get('last_seen_at', '') if face.get('last_seen_at') else ''
                    ])
            
            click.echo(f"\n✅ Exported {len(faces)} faces to CSV")
        
        else:
            click.echo(f"\n⚠️  Unsupported format: {format}")
            raise ValueError(f"Format must be 'json' or 'csv', got '{format}'")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.argument('query', required=False)
@click.option('--name', type=str, help='Filter by name')
@click.option('--face-id', type=int, help='Filter by face ID')
def search(query, name, face_id):
    """
    Search for faces in the database.
    
    Args:
        query: Search query (name or face_id)
        name: Filter by person name (partial match)
        face_id: Filter by exact face ID
    """
    click.echo("="*60)
    click.echo("🔍 SEARCH FACES")
    click.echo("="*60)
    
    if name:
        click.echo(f"Searching by name: {name}")
    if face_id:
            click.echo(f"Searching by ID: {face_id}")
    
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Get faces
        faces = pipeline.get_known_faces()
        
        if not faces:
            click.echo("\n⚠️  No faces in database")
            return
        
        # Filter faces
        results = []
        
        if name:
            # Partial name match (case-insensitive)
            query_lower = name.lower()
            for face in faces:
                if query_lower in face['name'].lower():
                    results.append(face)
        
        elif face_id:
            # Exact ID match
            for face in faces:
                if face['id'] == face_id:
                    results.append(face)
        
        # Display results
        if results:
            click.echo(f"Found {len(results)} matching face(s):")
            click.echo("-" * 80)
            
            for face in results:
                click.echo(
                    f"ID: {face['id']:<6} | "
                    f"Name: {face['name']:<30}"
                )
        else:
            click.echo("\n⚠️  No matching faces found")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--directory', type=click.Path(), required=True, help='Directory containing face images')
@click.option('--batch-size', type=int, default=10, help='Number of faces to process in batch')
@click.option('--output', '-o', type=click.Path(), help='Directory to save visualized images with bounding boxes')
def batch_enroll(directory, batch_size, output):
    """
    Batch enroll faces from a directory.
    
    Args:
        directory: Directory containing face images
        batch_size: Number of faces to process at once
        output: Directory to save visualized images with bounding boxes
    """
    click.echo("="*60)
    click.echo("📦 BATCH ENROLL FACES")
    click.echo("="*60)
    click.echo(f"Directory: {directory}")
    click.echo(f"Batch size: {batch_size}")
    if output:
        click.echo(f"Output: {output}")
    click.echo("")
    
    # Create output directory if specified
    output_dir = None
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Convert directory to Path object
        dir_path = Path(directory)
        
        # Get image files
        image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
        
        if not image_files:
            click.echo("\n⚠️  No image files found in directory")
            return
        
        click.echo(f"Found {len(image_files)} images")
        
        # Initialize stats tracking
        stats = {
            'total': len(image_files),
            'detected': 0,
            'enrolled': 0,
            'skipped': 0,
            'failed': 0
        }
        
        # Process in batches
        with click.progressbar(length=len(image_files), label='Enrolling faces') as bar:
            for i, image_file in enumerate(image_files):
                try:
                    # Extract name from filename
                    name = image_file.stem
                    click.echo(f"\n[{i+1}/{len(image_files)}] Processing: {name}")
                    
                    # Read and process image
                    import cv2
                    image = cv2.imread(str(image_file))
                    if image is None:
                        click.echo(f"   ✗ Failed to read image")
                        bar.update(1)
                        continue
                    
                    # Process frame to get detections and recognitions
                    results = pipeline.process_frame(image, recognize=True)
                    
                    # Only save images with faces detected
                    detected_faces = results.get('detections', [])
                    stats['detected'] += len(detected_faces)
                    
                    if output_dir and detected_faces:
                        # Visualize results
                        output_image = pipeline.visualize_results(image, results)
                        output_path = output_dir / f"{name}_result.jpg"
                        cv2.imwrite(str(output_path), output_image)
                        click.echo(f"   ✓ Saved: {output_path.name} ({len(detected_faces)} face(s))")
                    
                    # Only add face to database if face was detected
                    if detected_faces:
                        try:
                            face_id = pipeline.add_known_face_from_file(name, str(image_file))
                            
                            if face_id:
                                stats['enrolled'] += 1
                                click.echo(f"   ✓ Enrolled (ID: {face_id})")
                            else:
                                stats['failed'] += 1
                                click.echo(f"   ✗ Failed to enroll face")
                        except Exception as enroll_err:
                            stats['failed'] += 1
                            click.echo(f"   ✗ Failed: {str(enroll_err)}")
                    else:
                        stats['skipped'] += 1
                        click.echo(f"   ⊘ No face detected, skipped")
                    
                    # Small delay between batches
                    if (i + 1) % batch_size == 0:
                        import time
                        time.sleep(0.1)
                        
                except Exception as e:
                    stats['failed'] += 1
                    click.echo(f"   ✗ Failed: {str(e)}", err=True)
                    
                bar.update(1)
            
            # Final stats summary
            click.echo("")
            click.echo("="*60)
            click.echo("📊 BATCH ENROLLMENT SUMMARY")
            click.echo("="*60)
            click.echo(f"   Total images:     {stats['total']}")
            click.echo(f"   Faces detected:  {stats['detected']}")
            click.echo(f"   Faces enrolled:  {stats['enrolled']}")
            click.echo(f"   Faces skipped:   {stats['skipped']}")
            click.echo(f"   Failed:          {stats['failed']}")
            click.echo("="*60)
            
            if output_dir:
                click.echo(f"\n✓ Visualizations saved to: {output_dir}")
    
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--input', type=click.Path(), required=True, help='Input JSON or CSV file')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Input file format')
def import_faces(input, format):
    """
    Import faces from a JSON or CSV file.
    
    Args:
        input: Input file path
        format: Input file format (json or csv)
    """
    click.echo("="*60)
    click.echo("📥 IMPORT FACES")
    click.echo("="*60)
    click.echo(f"Input: {input}")
    click.echo(f"Format: {format.upper()}")
    click.echo("")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Import based on format
        if format == 'json':
            import json
            with open(input, 'r') as f:
                data = json.load(f)
                click.echo(f"Found {len(data)} faces to import")
            
            # Import faces
            imported = 0
            with click.progressbar(length=len(data), label='Importing') as bar:
                for i, face_data in enumerate(data):
                    try:
                        # Add face to database
                        face_id = pipeline.add_known_face(
                            name=face_data['name'],
                            metadata=face_data.get('metadata', {})
                        )
                        imported += 1
                        bar.update(1)
                    except Exception as e:
                        click.echo(f"   ✗ Failed: {str(e)}", err=True)
            
            click.echo(f"\n✅ Successfully imported {imported}/{len(data)} faces")
        
        elif format == 'csv':
            import csv
            with open(input, 'r', newline='') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            click.echo(f"Found {len(data)} faces to import")
            
            # Import faces
            imported = 0
            with click.progressbar(length=len(data), label='Importing') as bar:
                for i, row in enumerate(data):
                    try:
                        # Add face to database
                        face_id = pipeline.add_known_face(
                            name=row.get('name', ''),
                            metadata=row
                        )
                        imported += 1
                        bar.update(1)
                    except Exception as e:
                        click.echo(f"   ✗ Failed: {str(e)}", err=True)
            
            click.echo(f"\n✅ Successfully imported {imported}/{len(data)} faces")
        
        else:
            click.echo(f"\n⚠️  Unsupported format: {format}")
            raise ValueError(f"Format must be 'json' or 'csv', got '{format}'")
        
    except FileNotFoundError:
        click.echo(f"\n⚠️  Error: File not found: {input}")
        raise
    except json.JSONDecodeError:
        click.echo(f"\n⚠️  Error: Invalid JSON format in file")
        raise
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise


@click.command()
@click.option('--key', type=str, help='Configuration key')
@click.option('--value', type=str, help='Configuration value')
@click.option('--get', is_flag=True, help='Get a configuration value')
@click.option('--list', is_flag=True, help='List all configuration values')
def config(key, value, get, list):
    """
    Manage configuration settings.
    
    Args:
        key: Configuration key
        value: Configuration value to set
        get: Get a configuration value
        list: List all configuration values
    """
    click.echo("="*60)
    click.echo("⚙️ CONFIGURATION")
    click.echo("="*60)
    
    config_file = Path.home() / '.faceid'
    
    if list:
        # List all configuration
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
                click.echo("\nConfiguration:")
                for k, v in config_data.items():
                    click.echo(f"  {k}: {v}")
        else:
            click.echo("\n⚠️  No configuration file found")
    
    elif get:
        # Get specific configuration value
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            click.echo(f"\n{key}: {config_data.get(key, 'Not set')}")
        else:
            click.echo(f"\n⚠️  Configuration file not found")
    
    elif key and value:
        # Set configuration value
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config_data[key] = value
            
            # Save configuration
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            click.echo(f"\n✅ Set {key} = {value}")
        else:
            click.echo(f"\n⚠️  Configuration file not found")
    
    else:
        click.echo(f"\n⚠️  Please specify an operation (--get, --set, or --list)")


@click.command()
def help():
    """Show help information."""
    click.echo("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║         FACE RECOGNITION CLI - VERSION 0.1.0                ║
╚═══════════════════════════════════════════════╣
║  USAGE                                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                              ║
║  face-id start [--camera ID] [--output PATH]            ║
║                  [--no-display] [--save-faces PATH]        ║
║                                                              ║
║  face-id add-face --name NAME --image PATH              ║
║                    [--metadata JSON]                        ║
║                                                              ║
║  face-id list-faces                                       ║
║  face-id delete-face --face-id ID                           ║
║  face-id process-video --input PATH --output PATH     ║
║                  [--skip-frames N]                          ║
║                                                              ║
║  face-id stats                                             ║
║  face-id export-faces --output PATH [--format json|csv]    ║
║  face-id search [NAME | --face-id ID]                   ║
║  face-id batch-enroll --directory PATH [--batch-size N]        ║
║  face-id import-faces --input PATH [--format json|csv]      ║
║  face-id config --key KEY --value VALUE              ║
║                  [--get] [--list]                           ║
║                                                              ║
║  face-id help                                              ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                              ║
║  EXAMPLES                                                      ║
╠═══════════════════════════════════════════╣
║  Add a face and start recognition                         ║
║  $ face-id add-face --name "John Doe" --image john.jpg     ║
║  $ face-id start --camera 0 --save-faces ./detected  ║
║                                                              ║
║  List enrolled faces                                        ║
║  $ face-id list-faces                                         ║
║                                                              ║
║  Process a video file:                                       ║
║  $ face-id process-video --input video.mp4 --output result.mp4   ║
║                                                              ║
║  Export faces to JSON:                                         ║
║  $ face-id export-faces --output faces.json --format json     ║
║                                                              ║
║  Search for a face by name:                                     ║
║  $ face-id search --name "John"                              ║
║                                                              ║
║  Batch enroll multiple faces:                                    ║
║  $ face-id batch-enroll --directory ./photos --batch-size 20    ║
║                                                              ║
║                                                              ║
╚═════════════════════════════════════════════════════════════════════════╣
║                                                              ║
║  DISCLAIMER: This system is for educational and legitimate use only. ║
║  Always comply with local privacy laws and regulations.       ║
║                                                              ║
╚═══════════════════════════════╝
    """)



def main():
    """Main entry point for CLI."""
    # Add all commands to the CLI group
    cli.add_command(start)
    cli.add_command(add_face)
    cli.add_command(list_faces)
    cli.add_command(delete_face)
    cli.add_command(process_video)
    cli.add_command(export_faces)
    cli.add_command(search)
    cli.add_command(batch_enroll)
    cli.add_command(import_faces)
    cli.add_command(config)
    cli.add_command(help)
    cli()


if __name__ == '__main__':
    main()
