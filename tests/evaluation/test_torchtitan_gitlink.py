import os
from pathlib import Path

import pytest

os.environ.setdefault("LBT_LIGHT_IMPORT", "1")

import low_bits_training.analysis as _analysis_package

_ANALYSIS_PATH = Path(__file__).resolve().parents[2] / "low_bits_training/analysis"
if str(_ANALYSIS_PATH) not in _analysis_package.__path__:
    _analysis_package.__path__.insert(0, str(_ANALYSIS_PATH))

from low_bits_training.analysis import torchtitan_gitlink


def source_tree(root: Path) -> Path:
    package = root / "torchtitan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (root / torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_NAME).write_bytes(
        torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_BYTES
    )
    return root


def authority(path: Path) -> Path:
    path.write_bytes(torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_BYTES)
    return path


def test_source_export_accepts_exact_marker(tmp_path: Path) -> None:
    root = source_tree(tmp_path / "source")
    marker = authority(tmp_path / "authority")

    assert (
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )
        == torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT
    )


def test_source_export_rejects_marker_drift(tmp_path: Path) -> None:
    root = source_tree(tmp_path / "source")
    marker = authority(tmp_path / "authority")
    (root / torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_NAME).write_text(
        "0" * 40 + "\n", encoding="ascii"
    )

    with pytest.raises(RuntimeError, match="marker drift"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_checkout_rejects_wrong_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = source_tree(tmp_path / "source")
    marker = authority(tmp_path / "authority")
    monkeypatch.setattr(torchtitan_gitlink, "_git_head", lambda _: "0" * 40)

    with pytest.raises(RuntimeError, match="checkout commit drift"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )
