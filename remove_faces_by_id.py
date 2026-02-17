#!/usr/bin/env python3
"""
Remove Faces by ID Script
Remove specific faces from database by their ID.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from face_recognition.database import FaceDatabase
from face_recognition.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_faces() -> List[dict]:
    """List all faces in the database."""
    config = Config()
    db = FaceDatabase(str(config.DB_PATH))
    db.initialize()

    faces = db.list_faces()
    logger.info(f"Found {len(faces)} faces in database")

    return faces


def display_faces(faces: List[dict]) -> None:
    """Display faces in a table format."""
    if not faces:
        print("\n⚠️  No faces in database!")
        return

    print("\n" + "=" * 80)
    print(f"{'ID':<6} {'Name':<30} {'Created':<20} {'Seen':<10} {'Last Seen':<20}")
    print("-" * 80)

    for face in faces:
        face_id = face['id']
        name = face['name'][:28] + ('..' if len(face['name']) > 28 else face['name'])
        created = face.get('created_at', 'N/A')[:19]
        seen_count = face.get('seen_count', 0)
        last_seen = face.get('last_seen_at', 'Never')[:19] if face.get('last_seen_at') else 'Never'

        print(
            f"{face_id:<6} "
            f"{name:<30} "
            f"{created:<20} "
            f"{seen_count:<10} "
            f"{last_seen:<20}"
        )

    print("-" * 80)
    print(f"\nTotal: {len(faces)} faces\n")


def remove_faces(face_ids: List[int], force: bool = False) -> int:
    """Remove faces by their IDs.

    Args:
        face_ids: List of face IDs to remove
        force: If True, skip confirmation

    Returns:
        Number of faces successfully removed
    """
    if not face_ids:
        print("\n⚠️  No face IDs provided to remove!")
        return 0

    config = Config()
    db = FaceDatabase(str(config.DB_PATH))
    db.initialize()

    removed = 0
    failed = []

    print("\n" + "=" * 80)
    print("REMOVING FACES")
    print("=" * 80)

    for face_id in face_ids:
        # Get face details before deletion
        face = db.get_face(face_id)

        if not face:
            print(f"\n❌ Face ID {face_id} not found in database!")
            failed.append(face_id)
            continue

        print(f"\nRemoving: {face['name']} (ID: {face_id})")
        print(f"  Created: {face.get('created_at', 'N/A')[:19]}")
        print(f"  Seen: {face.get('seen_count', 0)} time(s)")

        # Delete face
        success = db.delete_face(face_id)

        if success:
            removed += 1
            print(f"  ✅ Removed successfully")
        else:
            print(f"  ❌ Failed to remove")
            failed.append(face_id)

    # Summary
    print("\n" + "=" * 80)
    print("REMOVAL SUMMARY")
    print("=" * 80)
    print(f"\nFaces removed: {removed}")
    print(f"Faces failed: {len(failed)}")

    if failed:
        print(f"\nFailed face IDs: {', '.join(map(str, failed))}")

    return removed


def remove_by_name(name: str, force: bool = False) -> int:
    """Remove all faces with a specific name.

    Args:
        name: Person name to remove (supports partial match)
        force: If True, skip confirmation

    Returns:
        Number of faces successfully removed
    """
    config = Config()
    db = FaceDatabase(str(config.DB_PATH))
    db.initialize()

    # Get all faces
    all_faces = db.list_faces()

    # Find faces matching name
    matching_faces = [
        f for f in all_faces
        if name.lower() in f['name'].lower()
    ]

    if not matching_faces:
        print(f"\n⚠️  No faces found matching '{name}'")
        return 0

    # Get IDs
    face_ids = [f['id'] for f in matching_faces]

    print(f"\nFound {len(matching_faces)} face(s) matching '{name}':")
    for face in matching_faces:
        print(f"  - {face['name']} (ID: {face['id']})")

    # Remove them
    return remove_faces(face_ids, force=force)


def remove_all(force: bool = False) -> int:
    """Remove ALL faces from database.

    Args:
        force: If True, skip confirmation

    Returns:
        Number of faces successfully removed
    """
    config = Config()
    db = FaceDatabase(str(config.DB_PATH))
    db.initialize()

    # Get count
    count = db.get_face_count()

    if count == 0:
        print("\n⚠️  Database is already empty!")
        return 0

    print(f"\n⚠️  WARNING: This will remove ALL {count} faces from the database!")
    print("This action cannot be undone!\n")

    if not force:
        try:
            confirm = input("Type 'DELETE ALL' to confirm: ").strip()
            if confirm != 'DELETE ALL':
                print("\nOperation cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n\nOperation cancelled.")
            return 0

    # Remove all faces
    removed = 0
    print("\n" + "=" * 80)
    print("REMOVING ALL FACES")
    print("=" * 80)

    # Delete all faces in one query
    db.connect()
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM faces")
    db.conn.commit()
    removed = count

    print(f"\n✅ Successfully removed all {count} faces from database!")
    return removed


def main():
    """Main function."""

    import argparse

    parser = argparse.ArgumentParser(
        description='Remove faces from database by ID or name',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all faces
  python remove_faces_by_id.py --list

  # Remove face by ID
  python remove_faces_by_id.py --id 10

  # Remove multiple faces by IDs
  python remove_faces_by_id.py --id 10 20 30

  # Remove faces by name (supports partial match)
  python remove_faces_by_id.py --name "John"

  # Remove all faces (dangerous!)
  python remove_faces_by_id.py --all --force

  # Remove face by ID without confirmation
  python remove_faces_by_id.py --id 10 --force
        """
    )

    parser.add_argument(
        '--id', '-i',
        type=int,
        nargs='+',
        help='Face ID(s) to remove (space-separated)'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        help='Remove faces by name (supports partial matching)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all faces in database'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Remove ALL faces from database (dangerous!)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              REMOVE FACES FROM DATABASE                          ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Handle different actions
    if args.list:
        # List all faces
        faces = list_faces()
        display_faces(faces)

    elif args.all:
        # Remove all faces
        removed = remove_all(force=args.force)
        if removed > 0:
            print("\n✅ Database cleared successfully!")

    elif args.name:
        # Remove faces by name
        removed = remove_by_name(args.name, force=args.force)
        if removed > 0:
            print(f"\n✅ Removed {removed} face(s) matching '{args.name}'")

    elif args.id:
        # Remove faces by ID(s)
        removed = remove_faces(args.id, force=args.force)
        if removed > 0:
            print(f"\n✅ Successfully removed {removed} face(s)!")
        else:
            print("\n⚠️  No faces were removed.")

    else:
        # No action specified - show help
        print("\nNo action specified. Use --help for usage information.")
        print("\nQuick start:")
        print("  python remove_faces_by_id.py --list    # List all faces")
        parser.print_help()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
