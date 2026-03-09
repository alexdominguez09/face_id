"""
Video Utilities Module
Helper functions for video codec handling and video operations.
"""

import cv2
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def get_fourcc_for_codec(codec_name: str, container: str = 'mp4') -> int:
    """
    Get FourCC code for given codec name and container.
    
    Args:
        codec_name: Codec name ('h264', 'h265', 'mp4v', 'xvid', 'avc1', etc.)
        container: Container format ('mp4', 'avi', 'mkv')
        
    Returns:
        FourCC code as integer
        
    Raises:
        ValueError: If codec/container combination is not supported
    """
    # Normalize inputs
    codec_name = codec_name.lower()
    container = container.lower()
    
    # Codec to FourCC mapping
    codec_map = {
        # H.264/AVC codecs
        'h264': {
            'mp4': cv2.VideoWriter_fourcc(*'avc1'),  # H.264 in MP4
            'avi': cv2.VideoWriter_fourcc(*'H264'),  # H.264 in AVI
            'mkv': cv2.VideoWriter_fourcc(*'H264'),  # H.264 in MKV
        },
        'avc1': {
            'mp4': cv2.VideoWriter_fourcc(*'avc1'),  # Alternative H.264
            'avi': cv2.VideoWriter_fourcc(*'H264'),
            'mkv': cv2.VideoWriter_fourcc(*'H264'),
        },
        'x264': {
            'mp4': cv2.VideoWriter_fourcc(*'avc1'),
            'avi': cv2.VideoWriter_fourcc(*'H264'),
            'mkv': cv2.VideoWriter_fourcc(*'H264'),
        },
        
        # H.265/HEVC codecs
        'h265': {
            'mp4': cv2.VideoWriter_fourcc(*'hevc'),  # H.265 in MP4
            'mkv': cv2.VideoWriter_fourcc(*'HEVC'),  # H.265 in MKV
        },
        'hevc': {
            'mp4': cv2.VideoWriter_fourcc(*'hevc'),
            'mkv': cv2.VideoWriter_fourcc(*'HEVC'),
        },
        
        # MPEG-4 codecs
        'mp4v': {
            'mp4': cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 in MP4
            'avi': cv2.VideoWriter_fourcc(*'XVID'),  # Fallback for AVI
        },
        
        # Other codecs
        'xvid': {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),  # XviD in AVI
            'mp4': cv2.VideoWriter_fourcc(*'mp4v'),  # Fallback for MP4
        },
        'mjpg': {
            'avi': cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG in AVI
        },
        'divx': {
            'avi': cv2.VideoWriter_fourcc(*'DIVX'),  # DivX in AVI
        },
    }
    
    # Check if codec is supported
    if codec_name not in codec_map:
        raise ValueError(f"Unsupported codec: {codec_name}")
    
    # Check if container is supported for this codec
    if container not in codec_map[codec_name]:
        # Try to find a fallback
        if container == 'mp4' and 'mp4v' in codec_map:
            logger.warning(f"Codec {codec_name} not supported in {container}, falling back to mp4v")
            return codec_map['mp4v']['mp4']
        elif container == 'avi' and 'xvid' in codec_map:
            logger.warning(f"Codec {codec_name} not supported in {container}, falling back to xvid")
            return codec_map['xvid']['avi']
        else:
            raise ValueError(f"Codec {codec_name} not supported in {container} container")
    
    return codec_map[codec_name][container]


def get_file_extension_for_container(container: str) -> str:
    """
    Get file extension for container format.
    
    Args:
        container: Container format ('mp4', 'avi', 'mkv')
        
    Returns:
        File extension with dot (e.g., '.mp4')
    """
    container = container.lower()
    
    extension_map = {
        'mp4': '.mp4',
        'avi': '.avi',
        'mkv': '.mkv',
        'mov': '.mov',
        'webm': '.webm',
    }
    
    if container not in extension_map:
        logger.warning(f"Unknown container {container}, using .mp4")
        return '.mp4'
    
    return extension_map[container]


def create_video_writer(output_path: str, 
                       width: int, 
                       height: int, 
                       fps: float,
                       codec: str = 'h264',
                       container: str = 'mp4',
                       quality: int = 95) -> cv2.VideoWriter:
    """
    Create a video writer with specified codec and settings.
    
    Args:
        output_path: Output file path
        width: Frame width
        height: Frame height
        fps: Frames per second
        codec: Video codec
        container: Container format
        quality: Video quality (0-100)
        
    Returns:
        OpenCV VideoWriter instance
        
    Raises:
        RuntimeError: If video writer cannot be created
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get FourCC code
    try:
        fourcc = get_fourcc_for_codec(codec, container)
    except ValueError as e:
        logger.error(f"Failed to get FourCC code: {e}")
        # Fall back to MP4V
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        logger.info("Falling back to MP4V codec")
    
    # Create video writer
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    
    # Set quality if supported (not all codecs support this)
    try:
        if hasattr(cv2, 'VIDEOWRITER_PROP_QUALITY'):
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
    except Exception as e:
        logger.debug(f"Could not set video quality: {e}")
    
    logger.info(f"Video writer created: {output_path} "
                f"(codec: {codec}, container: {container}, quality: {quality})")
    
    return writer


def get_video_properties(video_path: str) -> Dict:
    """
    Get video properties.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    properties = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'format': int(cap.get(cv2.CAP_PROP_FORMAT))
    }
    
    cap.release()
    
    # Convert FourCC code to string
    try:
        properties['codec_str'] = fourcc_to_string(properties['codec'])
    except:
        properties['codec_str'] = 'unknown'
    
    return properties


def fourcc_to_string(fourcc: int) -> str:
    """
    Convert FourCC code to string.
    
    Args:
        fourcc: FourCC code as integer
        
    Returns:
        FourCC code as string (e.g., 'avc1')
    """
    return ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


def string_to_fourcc(codec_str: str) -> int:
    """
    Convert string to FourCC code.
    
    Args:
        codec_str: FourCC code as string (e.g., 'avc1')
        
    Returns:
        FourCC code as integer
    """
    if len(codec_str) != 4:
        raise ValueError(f"FourCC code must be 4 characters, got: {codec_str}")
    
    return cv2.VideoWriter_fourcc(*codec_str)


def is_codec_available(codec: str, container: str = 'mp4') -> bool:
    """
    Check if codec is available in the current OpenCV build.
    
    Args:
        codec: Codec name
        container: Container format
        
    Returns:
        True if codec is available, False otherwise
    """
    try:
        fourcc = get_fourcc_for_codec(codec, container)
        
        # Test by trying to create a dummy writer
        test_path = '/tmp/test_codec_check.mp4'
        writer = cv2.VideoWriter(test_path, fourcc, 30, (640, 480))
        
        if writer.isOpened():
            writer.release()
            # Try to delete the test file
            try:
                Path(test_path).unlink(missing_ok=True)
            except:
                pass
            return True
        else:
            return False
            
    except Exception as e:
        logger.debug(f"Codec check failed: {e}")
        return False


def get_available_codecs() -> Dict[str, list]:
    """
    Get list of available codecs for different containers.
    
    Returns:
        Dictionary mapping containers to available codecs
    """
    available_codecs = {
        'mp4': [],
        'avi': [],
        'mkv': []
    }
    
    # Test common codecs
    test_codecs = [
        ('h264', 'avc1'),
        ('h265', 'hevc'),
        ('mp4v', 'mp4v'),
        ('xvid', 'XVID'),
        ('mjpg', 'MJPG'),
        ('divx', 'DIVX'),
    ]
    
    for codec_name, fourcc_str in test_codecs:
        for container in available_codecs.keys():
            try:
                if is_codec_available(codec_name, container):
                    available_codecs[container].append(codec_name)
            except:
                pass
    
    return available_codecs