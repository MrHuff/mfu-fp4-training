#!/usr/bin/env python3
"""Fail when first-party public documentation or route manifests dangle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")


def _local_target(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    elif " " in value:
        value = value.split(" ", 1)[0]
    value = unquote(value)
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return parsed.path


def _check_markdown(root: Path, relative: str) -> list[str]:
    path = root / relative
    if not path.is_file():
        return [relative]
    missing: list[str] = []
    for raw in MARKDOWN_LINK.findall(path.read_text(errors="strict")):
        target = _local_target(raw)
        if target is None:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            missing.append(f"{relative} -> outside-tree reference")
            continue
        if not resolved.exists():
            missing.append(f"{relative} -> {target}")
    return missing


def _check_inventory_code_paths(root: Path, relative: str) -> list[str]:
    """Validate path-like inline-code entries in a legal inventory."""

    path = root / relative
    if not path.is_file():
        return [relative]
    root_resolved = root.resolve()
    missing: list[str] = []
    for target in INLINE_CODE.findall(path.read_text(errors="strict")):
        if "/" not in target or any(character.isspace() for character in target):
            continue
        candidate = Path(target)
        if candidate.is_absolute() or ".." in candidate.parts:
            missing.append(f"{relative} -> outside-tree reference")
            continue
        if any(character in target for character in "*?["):
            if not list(root.glob(target)):
                missing.append(f"{relative} -> {target}")
            continue
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            missing.append(f"{relative} -> outside-tree reference")
            continue
        if not resolved.exists():
            missing.append(f"{relative} -> {target}")
    return missing


def _manifest_paths(root: Path) -> list[str]:
    required: set[str] = set()
    matrix_path = root / "release/route_component_matrix.json"
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text())
        required.update(matrix.get("shared_source_helpers", []))
        for component in matrix.get("kernel_components", {}).values():
            for key in ("source_root", "dependency"):
                value = component.get(key)
                if value:
                    required.add(value)
        for route in matrix.get("routes", []):
            for key in ("executable_config", "route_environment"):
                value = route.get(key)
                if value:
                    required.add(value)

    execution_path = root / "configs/paper/route_execution.json"
    if execution_path.is_file():
        execution = json.loads(execution_path.read_text())
        for key in ("base_config", "entrypoint", "external_input_contract"):
            value = execution.get(key)
            if value:
                required.add(value)
        for route in execution.get("routes", []):
            value = route.get("environment")
            if value:
                required.add(value)
    return sorted(required)


def check(root: Path) -> list[str]:
    missing: list[str] = []
    for relative in (
        "README.md",
        "configs/paper/README.md",
        "docs/technical_report/README.md",
    ):
        missing.extend(_check_markdown(root, relative))
    for relative in _manifest_paths(root):
        if not (root / relative).exists():
            missing.append(relative)
    missing.extend(_check_inventory_code_paths(root, "THIRD_PARTY_NOTICES.md"))
    return sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "release/components.json").is_file():
        print("public_reference_check=skipped_non_export_tree")
        return 0
    missing = check(root)
    if missing:
        for relative in missing:
            print(f"missing_public_reference={relative}")
        return 2
    print("public_reference_check=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
