"""Portable lookup for prebuilt Python extension modules."""

import importlib.machinery
import os


def find_existing_extension(label, roots, relpath):
    candidates = [os.path.join(root, relpath) for root in roots]

    filename = os.path.basename(relpath)
    module_name, marker, _ = filename.partition(".cpython-")
    if not marker and filename.endswith(".so"):
        module_name = filename[:-3]
    if module_name:
        relative_dir = os.path.dirname(relpath)
        for root in roots:
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                candidate = os.path.join(
                    root,
                    relative_dir,
                    module_name + suffix,
                )
                if candidate not in candidates:
                    candidates.append(candidate)

    for path in candidates:
        if os.path.isfile(path):
            return path
    raise AssertionError(f"{label} not found. Checked: {candidates}")
