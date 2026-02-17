#!/usr/bin/env python3
"""
Remove Duplicate Faces Script
Identifies and removes duplicate faces from the database.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

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


def find_duplicate_groups(
    faces: List[Dict],
    threshold: float
) -> List[List[Dict]]:
    """
    Group faces into duplicate clusters.

    Args:
        faces: List of all face data
        threshold: Similarity threshold to consider as duplicate

    Returns:
        List of duplicate groups (each group is a list of similar faces)
    """
    db = FaceDatabase()
    db.connect()

    duplicate_groups = []
    processed_ids = set()

    for face in faces:
        face_id = face['id']

        # Skip if already in a duplicate group
        if face_id in processed_ids:
            continue

        # Find all faces similar to this one
        group = [face]
        embedding = face['embedding']

        for other_face in faces:
            other_id = other_face['id']

            # Skip self and already processed
            if other_id <= face_id or other_id in processed_ids:
                continue

            # Calculate similarity
            similarity = FaceDatabase._cosine_similarity(
                embedding,
                other_face['embedding']
            )

            # If similar enough, add to group
            if similarity >= threshold:
                group.append(other_face)
                processed_ids.add(other_id)
                logger.debug(
                    f"Similar faces found: "
                    f"{face['name']} (ID:{face_id}) <-> "
                    f"{other_face['name']} (ID:{other_id}), "
                    f"similarity: {similarity:.3f}"
                )

        # Only keep groups with more than 1 face (actual duplicates)
        if len(group) > 1:
            duplicate_groups.append(group)
            processed_ids.update(f['id'] for f in group)

    return duplicate_groups


def process_duplicate_groups(
    duplicate_groups: List[List[Dict]],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Process duplicate groups to keep originals and delete duplicates.

    Args:
        duplicate_groups: List of duplicate face groups
        dry_run: If True, don't actually delete (just show what would happen)

    Returns:
        Tuple of (faces_kept, faces_deleted)
    """
    db = FaceDatabase()

    total_kept = 0
    total_deleted = 0

    print("\n" + "=" * 70)
    print("PROCESSING DUPLICATE GROUPS")
    print("=" * 70)

    for i, group in enumerate(duplicate_groups, 1):
        # Sort by ID (keep the earliest one)
        group_sorted = sorted(group, key=lambda f: f['id'])
        original = group_sorted[0]
        duplicates = group_sorted[1:]

        print(f"\n--- Duplicate Group {i} ---")
        print(f"KEEPING: {original['name']} (ID: {original['id']})")
        print(f"         Created: {original.get('created_at', 'N/A')[:19]}")
        print(f"DELETING {len(duplicates)} duplicate(s):")

        for duplicate in duplicates:
            print(f"  - {duplicate['name']} (ID: {duplicate['id']})")
            print(f"    Created: {duplicate.get('created_at', 'N/A')[:19]}")
            print(f"    Seen {duplicate.get('seen_count', 0)} time(s)")

            if not dry_run:
                # Actually delete the duplicate
                success = db.delete_face(duplicate['id'])
                if success:
                    total_deleted += 1
                else:
                    logger.error(f"Failed to delete face ID {duplicate['id']}")

        total_kept += 1

    return total_kept, total_deleted


def main():
    """Main function to remove duplicate faces."""

    import argparse
    parser = argparse.ArgumentParser(description='Remove duplicate faces from database')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt and remove all duplicates')
    parser.add_argument('--list', '-l', action='store_true',
                       help='Only list duplicates, do not remove them')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                       help='Override duplicate detection threshold (default: 0.85)')
    args = parser.parse_args()

    print()
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║              REMOVE DUPLICATE FACES FROM DATABASE                  ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    print()

    # Get configuration
    config = Config()
    duplicate_threshold = args.threshold or getattr(config, 'DUPLICATE_THRESHOLD', 0.85)

    print(f"Duplicate detection threshold: {duplicate_threshold}")
    print(f"(Faces with similarity ≥ {duplicate_threshold} are considered duplicates)")
    print()

    # Initialize database
    db = FaceDatabase()
    db.initialize()

    # Get all faces
    print("Loading all faces from database...")
    all_faces = db.list_faces()
    print(f"Total faces in database: {len(all_faces)}")
    print()

    if len(all_faces) == 0:
        print("No faces in database. Nothing to do.")
        return

    # Find duplicate groups
    print("Analyzing faces for duplicates...")
    print("(This may take a while for large databases...)")
    print()

    duplicate_groups = find_duplicate_groups(all_faces, duplicate_threshold)

    if not duplicate_groups:
        print("\n✅ No duplicates found in database!")
        print("All faces are unique.")
        return

    # Show summary of duplicates found
    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS COMPLETE")
    print("=" * 70)

    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
    total_groups = len(duplicate_groups)

    print(f"\nFound {total_groups} duplicate group(s)")
    print(f"Total duplicates to remove: {total_duplicates}")
    print(f"Unique faces that will remain: {len(all_faces) - total_duplicates}")
    print()

    # Ask for confirmation (or skip if --yes flag)
    print("=" * 70)
    if args.list:
        print("\n📋 LISTING DUPLICATES ONLY (NO DELETION)")
        faces_kept, faces_deleted = process_duplicate_groups(
            duplicate_groups,
            dry_run=True
        )
        print("\n✅ Listing complete. Use --yes to actually remove duplicates.")
        return

    if not args.yes:
        try:
            response = input(
                f"Remove {total_duplicates} duplicate face(s)? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nOperation cancelled by user.")
            return

        if response not in ['y', 'yes']:
            print("\nOperation cancelled. No faces were deleted.")
            return

    # Process duplicates
    faces_kept, faces_deleted = process_duplicate_groups(
        duplicate_groups,
        dry_run=False
    )

    # Final summary
    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"\n✅ Faces kept: {faces_kept}")
    print(f"✅ Faces deleted: {faces_deleted}")
    print(f"✅ Database size reduced by: {faces_deleted} faces")
    print(f"✅ Final database size: {len(all_faces) - faces_deleted} unique faces")
    print()

    # Verify database
    remaining_count = db.get_face_count()
    print(f"Verification: {remaining_count} faces in database")
    print()

    if remaining_count == len(all_faces) - faces_deleted:
        print("✅ Database cleanup verified successfully!")
    else:
        print("⚠️  Warning: Database count doesn't match expected!")


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
