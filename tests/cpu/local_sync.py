#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

#!/usr/bin/env python3
"""
Local file sync that mimics aws s3 sync output format. Useful for mocking locally `aws s3 sync` for unit tests.

Usage: python local_sync.py <source> <destination> [--dryrun]
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple


def get_relative_files(directory: Path) -> dict[str, float]:
    """
    Get all files in directory with their modification times.

    Returns:
        Dict mapping relative paths to modification times
    """
    files = {}
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = Path(root) / filename
            rel_path = filepath.relative_to(directory)
            files[str(rel_path)] = filepath.stat().st_mtime
    return files


def sync_directories(
    source: Path, dest: Path, no_progress: bool = True, dryrun: bool = False
) -> Tuple[int, int, int]:
    """
    Sync source directory to destination, mimicking aws s3 sync output.

    Returns:
        Tuple of (uploaded, downloaded, deleted) counts
    """
    source = source.resolve()
    dest = dest.resolve()
    source.mkdir(exist_ok=True)

    # Get file lists
    source_files = get_relative_files(source)
    dest_files = get_relative_files(dest) if dest.exists() else {}
    # assert len(dest_files) == 0, "Destination directory needs to be empty."

    uploaded = 0
    # Copy new or modified files (upload equivalent)
    for rel_path, source_mtime in source_files.items():
        source_file = source / rel_path
        dest_file = dest / rel_path

        # Check if file needs to be copied
        should_copy = False
        if rel_path not in dest_files:
            should_copy = True  # New file
        elif source_mtime > dest_files[rel_path]:
            should_copy = True  # Modified file

        if should_copy:
            prefix = "(dryrun) " if dryrun else ""
            print(f"{prefix}upload: {source_file} to {dest_file}")
            if not dryrun:
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_file)
            uploaded += 1
    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Sync local directories with aws s3 sync-like output"
    )
    parser.add_argument("source", help="Source directory")
    parser.add_argument("destination", help="Destination directory")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Deactivate progress bar."
    )
    args = parser.parse_args()

    try:
        source = Path(args.source)
        dest = Path(args.destination)
        sync_directories(source, dest, no_progress=args.no_progress, dryrun=args.dryrun)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
