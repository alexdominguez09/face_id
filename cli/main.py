"""
CLI Main Module
Command-line interface for face recognition system.
"""

import click
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition import RecognitionPipeline, VideoProcessor, Config


@click.group()
def cli():
    """Face Recognition CLI - Real-time face identification system."""
    pass


@cli.command()
@click.option('--camera', default=0, help='Camera device number')
@click.option('--output', help='Output video file path')
@click.option('--no-display', is_flag=True, help='Disable video display')
def start(camera, output, no_display):
    """
    Start real-time face detection and recognition.
    
    Args:
        camera: Camera device number
        output: Optional output video file
        no_display: Disable video display window
    """
    click.echo(f"Starting face recognition on camera {camera}")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Create video processor
        processor = VideoProcessor(pipeline=pipeline, config=config)
        
        # Process camera
        processor.process_camera(
            camera_id=camera,
            display=not no_display,
            output_file=output
        )
        
    except KeyboardInterrupt:
        click.echo("\nStopping...")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
@click.option('--name', required=True, help='Person name')
@click.option('--image', required=True, help='Image file path')
def add_face(name, image):
    """
    Add a new face to the database.
    
    Args:
        name: Person name
        image: Image file path
    """
    click.echo(f"Adding face for {name} from {image}")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Add face
        face_id = pipeline.add_known_face_from_file(name, image)
        
        click.echo(f"✓ Face added successfully (ID: {face_id})")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
def list_faces():
    """List all enrolled faces."""
    click.echo("Listing faces...")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Get faces
        faces = pipeline.get_known_faces()
        
        if not faces:
            click.echo("No faces in database")
            return
        
        click.echo(f"\nTotal faces: {len(faces)}\n")
        click.echo(f"{'ID':<5} {'Name':<30} {'Created':<20} {'Seen Count'}")
        click.echo("-" * 70)
        
        for face in faces:
            click.echo(f"{face['id']:<5} {face['name']:<30} "
                      f"{face['created_at']:<20} {face.get('seen_count', 0)}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
@click.option('--face-id', required=True, help='Face ID to delete')
def delete_face(face_id):
    """
    Delete a face from the database.
    
    Args:
        face_id: Face ID to delete
    """
    click.echo(f"Deleting face {face_id}")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Delete face
        deleted = pipeline.remove_known_face(int(face_id))
        
        if deleted:
            click.echo(f"✓ Face {face_id} deleted successfully")
        else:
            click.echo(f"✗ Face {face_id} not found")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
@click.option('--input', required=True, help='Input video file path')
@click.option('--output', required=True, help='Output video file path')
@click.option('--skip-frames', default=0, help='Number of frames to skip between processing')
def process_video(input, output, skip_frames):
    """
    Process a video file for face recognition.
    
    Args:
        input: Input video file path
        output: Output video file path
        skip_frames: Number of frames to skip between processing
    """
    click.echo(f"Processing video: {input} -> {output}")
    
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        # Create video processor
        processor = VideoProcessor(pipeline=pipeline, config=config)
        
        # Process video
        stats = processor.process_video_file(
            input_path=input,
            output_path=output,
            skip_frames=skip_frames
        )
        
        # Print statistics
        click.echo("\n" + "="*50)
        click.echo("Processing Statistics")
        click.echo("="*50)
        click.echo(f"Total frames: {stats['total_frames']}")
        click.echo(f"Processed frames: {stats['processed_frames']}")
        click.echo(f"Elapsed time: {stats['elapsed_time']:.2f}s")
        click.echo(f"Average FPS: {stats['avg_fps']:.2f}")
        click.echo(f"Known faces: {stats['pipeline_stats']['known_faces']}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
def stats():
    """Show database statistics."""
    try:
        # Initialize pipeline
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Get stats
        statistics = pipeline.get_stats()
        
        click.echo("\n" + "="*50)
        click.echo("Database Statistics")
        click.echo("="*50)
        click.echo(f"Known faces: {statistics['known_faces']}")
        click.echo(f"Database path: {config.DB_PATH}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
