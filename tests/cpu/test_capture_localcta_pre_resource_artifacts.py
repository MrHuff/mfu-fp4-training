from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "scripts/nvl72/prepare_fp4_head_b300_runtime.sh"
CONTRACT = (
    REPO_ROOT / "scripts/nvl72/localcta_pre_resource_publication_v3_contract.json"
)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import capture_localcta_pre_resource_artifacts as capture  # noqa: E402
import verify_localcta_pre_resource_capture as verifier  # noqa: E402


RESOURCE_STDOUT = b"raw resource output\x00REG:64 STACK:432\n"
RESOURCE_STDERR = b"raw resource stderr\n"
ELF_STDOUT = b"raw ELF output\x00STT_FUNC production\n"
ELF_STDERR = b"raw ELF stderr\n"
MODULE_BYTES = b"exact-localcta-extension\x00\xff" * 1024


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_cuobjdump(path: Path) -> Path:
    """Make cuobjdump prove that the passed /proc/self/fd is inherited/readable."""

    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "option, module = sys.argv[1:]\n"
        "if not module.startswith('/proc/self/fd/'):\n"
        "    raise SystemExit(17)\n"
        "if not os.readlink(module).endswith('.so'):\n"
        "    raise SystemExit(18)\n"
        "with open(module, 'rb') as stream:\n"
        "    if not stream.read():\n"
        "        raise SystemExit(20)\n"
        "if option == '--dump-resource-usage':\n"
        f"    sys.stdout.buffer.write({RESOURCE_STDOUT!r})\n"
        f"    sys.stderr.buffer.write({RESOURCE_STDERR!r})\n"
        "elif option == '--dump-elf-symbols':\n"
        f"    sys.stdout.buffer.write({ELF_STDOUT!r})\n"
        f"    sys.stderr.buffer.write({ELF_STDERR!r})\n"
        "else:\n"
        "    raise SystemExit(19)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _inputs(tmp_path: Path, name: str = "inputs") -> tuple[Path, Path]:
    source_dir = tmp_path / name
    source_dir.mkdir()
    module = source_dir / "_tk_quant_localcta_v4.cpython-312-x86_64-linux-gnu.so"
    module.write_bytes(MODULE_BYTES)
    return module, _fake_cuobjdump(source_dir / "cuobjdump")


def _make_removable(directory: Path) -> None:
    if os.path.lexists(directory) and not directory.is_symlink():
        try:
            os.chmod(directory, 0o700, follow_symlinks=False)
        except FileNotFoundError:
            pass


def _assert_incomplete(target: Path) -> None:
    assert target.is_dir() and not target.is_symlink()
    assert _mode(target) == 0o700
    assert not os.path.lexists(target / capture.PUBLICATION)


def _assert_sealed(target: Path) -> None:
    assert target.is_dir() and not target.is_symlink()
    assert _mode(target) == 0o555
    assert (target / capture.PUBLICATION).is_file()
    assert _mode(target / capture.PUBLICATION) == 0o444


def _publish(tmp_path: Path, name: str = "capture") -> tuple[Path, Path, dict]:
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / name
    receipt = capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    return module, target, receipt


def _rewrite_member(target: Path, name: str, data: bytes) -> None:
    os.chmod(target, 0o700)
    member = target / name
    os.chmod(member, 0o600, follow_symlinks=False)
    member.write_bytes(data)
    os.chmod(member, 0o444, follow_symlinks=False)
    os.chmod(target, 0o555)


def test_success_is_exact_fd_bound_v3_and_runs_both_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    events: list[str] = []
    real_payload_verify = capture.verify_capture_payload_fd
    real_final_verify = capture.verify_capture_fd
    real_run = capture.subprocess.run

    def checked_payload_verify(**kwargs):
        events.append("payload")
        assert stat.S_IMODE(os.fstat(kwargs["target_fd"]).st_mode) == 0o700
        assert capture.PUBLICATION not in os.listdir(kwargs["target_fd"])
        return real_payload_verify(**kwargs)

    def checked_final_verify(**kwargs):
        events.append("complete")
        assert stat.S_IMODE(os.fstat(kwargs["target_fd"]).st_mode) == 0o555
        assert capture.PUBLICATION in os.listdir(kwargs["target_fd"])
        return real_final_verify(**kwargs)

    def checked_subprocess_run(command, **kwargs):
        assert command[2].startswith("/proc/self/fd/")
        assert kwargs["pass_fds"] == (int(command[2].rsplit("/", 1)[1]),)
        assert kwargs["stdin"] is capture.subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert kwargs["shell"] is False
        assert kwargs["timeout"] > 0
        return real_run(command, **kwargs)

    monkeypatch.setattr(capture, "verify_capture_payload_fd", checked_payload_verify)
    monkeypatch.setattr(capture, "verify_capture_fd", checked_final_verify)
    monkeypatch.setattr(capture.subprocess, "run", checked_subprocess_run)
    try:
        receipt = capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
        # The complete verifier runs once through the original target-parent
        # dirfd and once more through the freshly re-walked parent chain.
        assert events == ["payload", "complete", "complete"]
        assert receipt == verifier.verify_capture_path(target)
        _assert_sealed(target)

        expected = {
            module.name,
            capture.RESOURCE_STDOUT,
            capture.RESOURCE_STDERR,
            capture.ELF_STDOUT,
            capture.ELF_STDERR,
            capture.RECEIPT,
            capture.CHECKSUMS,
            capture.PUBLICATION,
        }
        assert set(os.listdir(target)) == expected
        assert all(_mode(target / name) == 0o444 for name in expected)
        assert (target / module.name).read_bytes() == MODULE_BYTES
        assert (target / capture.RESOURCE_STDOUT).read_bytes() == RESOURCE_STDOUT
        assert (target / capture.RESOURCE_STDERR).read_bytes() == RESOURCE_STDERR
        assert (target / capture.ELF_STDOUT).read_bytes() == ELF_STDOUT
        assert (target / capture.ELF_STDERR).read_bytes() == ELF_STDERR

        on_disk_receipt = json.loads((target / capture.RECEIPT).read_bytes())
        publication = json.loads((target / capture.PUBLICATION).read_bytes())
        assert on_disk_receipt == receipt
        assert receipt["schema"] == capture.CAPTURE_SCHEMA
        assert receipt["module"]["bytes"] == len(MODULE_BYTES)
        assert receipt["module"]["sha256"] == hashlib.sha256(MODULE_BYTES).hexdigest()
        assert publication["schema"] == capture.PUBLICATION_SCHEMA
        assert publication["state"] == "complete"
        assert publication["capture_schema"] == capture.CAPTURE_SCHEMA
        target_stat = os.stat(target, follow_symlinks=False)
        assert publication["target"] == {
            "device": target_stat.st_dev,
            "inode": target_stat.st_ino,
        }
        assert publication["capture_receipt"] == {
            "file": capture.RECEIPT,
            "sha256": _sha256(target / capture.RECEIPT),
        }
        assert publication["capture_ledger"] == {
            "file": capture.CHECKSUMS,
            "sha256": _sha256(target / capture.CHECKSUMS),
        }
    finally:
        _make_removable(target)


@pytest.mark.parametrize("target", ["capture", "a/../capture"])
def test_rejects_relative_and_lexically_unsafe_targets(tmp_path: Path, target: str):
    module, cuobjdump = _inputs(tmp_path)
    with pytest.raises(ValueError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    with pytest.raises(ValueError):
        capture.capture_artifacts(
            module,
            tmp_path / "parent" / ".." / "capture",
            cuobjdump=cuobjdump,
        )


@pytest.mark.parametrize("target_kind", ["file", "directory", "symlink", "broken"])
def test_existing_target_is_never_reused_or_modified(tmp_path: Path, target_kind: str):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    if target_kind == "file":
        target.write_bytes(b"owner-file")
    elif target_kind == "directory":
        target.mkdir(mode=0o751)
        (target / "owner-data").write_bytes(b"owner-directory")
    elif target_kind == "symlink":
        real_target = tmp_path / "real-target"
        real_target.mkdir()
        target.symlink_to(real_target, target_is_directory=True)
    else:
        target.symlink_to(tmp_path / "absent-target", target_is_directory=True)
    before_link = os.readlink(target) if target.is_symlink() else None
    before_mode = _mode(target) if not target.is_symlink() else None

    with pytest.raises(FileExistsError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)

    if target_kind == "file":
        assert target.read_bytes() == b"owner-file"
    elif target_kind == "directory":
        assert (target / "owner-data").read_bytes() == b"owner-directory"
        assert _mode(target) == before_mode
    else:
        assert target.is_symlink() and os.readlink(target) == before_link


def test_missing_and_symlinked_ancestors_and_source_symlink_are_rejected(
    tmp_path: Path,
):
    module, cuobjdump = _inputs(tmp_path)
    with pytest.raises(ValueError):
        capture.capture_artifacts(
            module, tmp_path / "missing" / "capture", cuobjdump=cuobjdump
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        capture.capture_artifacts(
            module, linked_parent / "capture", cuobjdump=cuobjdump
        )

    source_link = tmp_path / "linked-module.so"
    source_link.symlink_to(module)
    with pytest.raises(ValueError):
        capture.capture_artifacts(
            source_link, tmp_path / "capture-a", cuobjdump=cuobjdump
        )

    linked_source_parent = tmp_path / "linked-inputs"
    linked_source_parent.symlink_to(module.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        capture.capture_artifacts(
            linked_source_parent / module.name,
            tmp_path / "capture-b",
            cuobjdump=cuobjdump,
        )


def test_two_concurrent_publishers_have_exactly_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    barrier = threading.Barrier(2)
    real_claim = capture._claim_target

    def synchronized_claim(parent_fd: int, target_name: str):
        barrier.wait(timeout=10)
        return real_claim(parent_fd, target_name)

    monkeypatch.setattr(capture, "_claim_target", synchronized_claim)

    def publish():
        try:
            return capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
        except BaseException as error:  # Return it so both workers always join.
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: publish(), range(2)))
        assert sum(isinstance(result, dict) for result in results) == 1
        assert sum(isinstance(result, FileExistsError) for result in results) == 1
        verifier.verify_capture_path(target)
    finally:
        _make_removable(target)


def test_destination_entry_injection_cannot_overwrite_attacker_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_claim = capture._claim_target
    sentinel = b"attacker-owned-destination"

    def claim_then_inject(parent_fd: int, target_name: str):
        target_fd, metadata = real_claim(parent_fd, target_name)
        injected_fd = os.open(
            module.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=target_fd,
        )
        try:
            os.write(injected_fd, sentinel)
        finally:
            os.close(injected_fd)
        return target_fd, metadata

    monkeypatch.setattr(capture, "_claim_target", claim_then_inject)
    with pytest.raises(FileExistsError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    assert (target / module.name).read_bytes() == sentinel
    _assert_incomplete(target)


def test_in_place_source_mutation_is_detected_before_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_run = capture._run_cuobjdump
    calls = 0

    def mutate_after_first_capture(executable, option, module_fd, module_name):
        nonlocal calls
        result = real_run(executable, option, module_fd, module_name)
        calls += 1
        if calls == 1:
            module.write_bytes(b"changed-held-source")
        return result

    monkeypatch.setattr(capture, "_run_cuobjdump", mutate_after_first_capture)
    with pytest.raises(RuntimeError, match="source changed|bound localCTA source"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_incomplete(target)


def test_source_path_replacement_after_seal_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    original = module.with_suffix(module.suffix + ".original")
    real_verify = capture.verify_capture_fd
    injected = False

    def verify_then_replace(**kwargs):
        nonlocal injected
        result = real_verify(**kwargs)
        if not injected:
            injected = True
            module.rename(original)
            module.write_bytes(MODULE_BYTES)
        return result

    monkeypatch.setattr(capture, "verify_capture_fd", verify_then_replace)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_sealed(target)
    assert (
        os.stat(module, follow_symlinks=False).st_ino
        != os.stat(original, follow_symlinks=False).st_ino
    )


@pytest.mark.parametrize("nth_write", range(1, 9))
def test_every_nth_direct_write_failure_leaves_only_an_incomplete_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nth_write: int
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_write = capture._write_all
    writes = 0

    def fail_nth(file_descriptor: int, data: bytes):
        nonlocal writes
        writes += 1
        if writes == nth_write:
            raise OSError(errno.EIO, f"simulated write {nth_write}")
        return real_write(file_descriptor, data)

    monkeypatch.setattr(capture, "_write_all", fail_nth)
    with pytest.raises(OSError, match="simulated write"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_incomplete(target)


@pytest.mark.parametrize("failed_invocation", [1, 2])
def test_each_cuobjdump_failure_is_pre_marker_and_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_invocation: int
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_run = capture._run_cuobjdump
    calls = 0

    def fail_selected(executable, option, module_fd, module_name):
        nonlocal calls
        calls += 1
        if calls == failed_invocation:
            raise RuntimeError(f"simulated cuobjdump {calls}")
        return real_run(executable, option, module_fd, module_name)

    monkeypatch.setattr(capture, "_run_cuobjdump", fail_selected)
    with pytest.raises(RuntimeError, match="simulated cuobjdump"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_incomplete(target)


def test_marker_rollback_failure_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_write = capture._write_all
    real_unlink = os.unlink

    def fail_marker_write(file_descriptor: int, data: bytes):
        if capture.PUBLICATION_SCHEMA.encode() in data:
            raise OSError(errno.EIO, "simulated marker write")
        return real_write(file_descriptor, data)

    def fail_marker_rollback(path, *args, **kwargs):
        if os.fspath(path) == capture.PUBLICATION:
            raise OSError(errno.EIO, "simulated marker rollback")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(capture, "_write_all", fail_marker_write)
    monkeypatch.setattr(os, "unlink", fail_marker_rollback)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    assert target.is_dir() and _mode(target) == 0o700
    assert os.path.lexists(target / capture.PUBLICATION)


def test_interrupt_after_marker_open_but_before_callback_never_leaves_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Cover the async window between open(O_EXCL) and marker bookkeeping."""

    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_create = capture._create_readonly_bytes

    def open_marker_then_interrupt(
        directory_fd: int,
        name: str,
        data: bytes,
        *,
        on_open=None,
    ):
        if name != capture.PUBLICATION:
            return real_create(
                directory_fd,
                name,
                data,
                on_open=on_open,
            )
        marker_fd = os.open(
            name,
            capture._regular_create_flags(),
            0o600,
            dir_fd=directory_fd,
        )
        os.close(marker_fd)
        raise KeyboardInterrupt("simulated interruption before marker callback")

    monkeypatch.setattr(capture, "_create_readonly_bytes", open_marker_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="before marker callback"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_incomplete(target)


def test_marker_rollback_directory_fsync_failure_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_write = capture._write_all
    real_fsync = capture._fsync_fd

    def fail_marker_write(file_descriptor: int, data: bytes):
        if capture.PUBLICATION_SCHEMA.encode() in data:
            raise OSError(errno.EIO, "simulated marker write")
        return real_write(file_descriptor, data)

    def fail_rollback_fsync(file_descriptor: int, boundary: str):
        if boundary == "marker-rollback-target-directory":
            raise OSError(errno.EIO, "simulated rollback fsync")
        return real_fsync(file_descriptor, boundary)

    monkeypatch.setattr(capture, "_write_all", fail_marker_write)
    monkeypatch.setattr(capture, "_fsync_fd", fail_rollback_fsync)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    assert target.is_dir() and _mode(target) == 0o700
    assert not os.path.lexists(target / capture.PUBLICATION)


@pytest.mark.parametrize("effect", ["before", "after"])
def test_target_seal_chmod_boundary_is_classified_by_observed_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect: str
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_fchmod = os.fchmod

    def fail_target_seal(file_descriptor: int, mode: int):
        metadata = os.fstat(file_descriptor)
        if stat.S_ISDIR(metadata.st_mode) and mode == 0o555:
            if effect == "after":
                real_fchmod(file_descriptor, mode)
            raise OSError(errno.EIO, f"simulated chmod {effect}")
        return real_fchmod(file_descriptor, mode)

    monkeypatch.setattr(os, "fchmod", fail_target_seal)
    expected = capture.IndeterminatePublicationError if effect == "after" else OSError
    with pytest.raises(expected, match="indeterminate|simulated chmod"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    if effect == "after":
        _assert_sealed(target)
    else:
        _assert_incomplete(target)


@pytest.mark.parametrize(
    ("boundary", "indeterminate"),
    [
        ("pre-marker-target-directory", False),
        ("sealed-target-directory", True),
        ("sealed-parent-directory", True),
    ],
)
def test_directory_fsync_boundaries_have_explicit_publication_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    indeterminate: bool,
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    real_fsync = capture._fsync_fd

    def fail_boundary(file_descriptor: int, current_boundary: str):
        if current_boundary == boundary:
            raise OSError(errno.EIO, f"simulated fsync {boundary}")
        return real_fsync(file_descriptor, current_boundary)

    monkeypatch.setattr(capture, "_fsync_fd", fail_boundary)
    expected = capture.IndeterminatePublicationError if indeterminate else OSError
    with pytest.raises(expected):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    if indeterminate:
        _assert_sealed(target)
    else:
        _assert_incomplete(target)


def test_final_verifier_failure_is_always_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"

    def fail_verifier(**_kwargs):
        raise RuntimeError("simulated strict verifier failure")

    monkeypatch.setattr(capture, "verify_capture_fd", fail_verifier)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_sealed(target)


@pytest.mark.parametrize("chain", ["target", "source"])
def test_final_fresh_chain_reopen_detects_ancestor_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chain: str
):
    module, cuobjdump = _inputs(tmp_path, "source-parent")
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    target = output_parent / "capture"
    old_parent = tmp_path / f"old-{chain}-parent"
    real_verify = capture.verify_capture_fd
    injected = False

    def verify_then_rebind(**kwargs):
        nonlocal injected
        result = real_verify(**kwargs)
        if injected:
            return result
        injected = True
        if chain == "target":
            output_parent.rename(old_parent)
            output_parent.mkdir()
        else:
            module.parent.rename(old_parent)
            module.parent.mkdir()
            (module.parent / module.name).write_bytes(MODULE_BYTES)
        return result

    monkeypatch.setattr(capture, "verify_capture_fd", verify_then_rebind)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)

    sealed_target = (
        old_parent / "capture" if chain == "target" else output_parent / "capture"
    )
    _assert_sealed(sealed_target)
    _make_removable(sealed_target)


def test_close_error_after_sealed_success_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    assert hasattr(capture, "_close_fd"), "capture must expose its close fault boundary"
    real_close = capture._close_fd
    failed = False

    def close_then_fail_once(file_descriptor: int):
        nonlocal failed
        metadata = os.fstat(file_descriptor)
        if (
            not failed
            and stat.S_ISDIR(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o555
        ):
            failed = True
            real_close(file_descriptor)
            raise OSError(errno.EIO, "simulated final close")
        return real_close(file_descriptor)

    monkeypatch.setattr(capture, "_close_fd", close_then_fail_once)
    with pytest.raises(capture.IndeterminatePublicationError):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    _assert_sealed(target)


def _mutate_completed_capture(target: Path, module_name: str, mutation: str) -> None:
    if mutation == "extra-entry":
        os.chmod(target, 0o700)
        (target / "unexpected").write_bytes(b"injected")
        os.chmod(target / "unexpected", 0o444)
        os.chmod(target, 0o555)
    elif mutation == "member-mode":
        os.chmod(target / capture.RESOURCE_STDOUT, 0o644)
    elif mutation == "target-mode":
        os.chmod(target, 0o700)
    elif mutation == "marker-duplicate-key":
        _rewrite_member(target, capture.PUBLICATION, b'{"schema":1,"schema":2}\n')
    elif mutation == "marker-nonfinite":
        _rewrite_member(target, capture.PUBLICATION, b'{"schema":NaN}\n')
    elif mutation == "marker-hash":
        marker = json.loads((target / capture.PUBLICATION).read_bytes())
        marker["capture_receipt"]["sha256"] = "0" * 64
        _rewrite_member(
            target,
            capture.PUBLICATION,
            (json.dumps(marker, sort_keys=True) + "\n").encode(),
        )
    elif mutation == "marker-target-inode":
        marker = json.loads((target / capture.PUBLICATION).read_bytes())
        marker["target"]["inode"] += 1
        _rewrite_member(
            target,
            capture.PUBLICATION,
            (json.dumps(marker, sort_keys=True) + "\n").encode(),
        )
    elif mutation == "receipt-schema":
        receipt = json.loads((target / capture.RECEIPT).read_bytes())
        receipt["schema"] = "localcta-pre-resource-capture-v2"
        _rewrite_member(
            target,
            capture.RECEIPT,
            (json.dumps(receipt, sort_keys=True) + "\n").encode(),
        )
    elif mutation == "uppercase-sha":
        receipt = json.loads((target / capture.RECEIPT).read_bytes())
        receipt["module"]["sha256"] = receipt["module"]["sha256"].upper()
        _rewrite_member(
            target,
            capture.RECEIPT,
            (json.dumps(receipt, sort_keys=True) + "\n").encode(),
        )
    elif mutation == "ledger":
        ledger = (target / capture.CHECKSUMS).read_bytes()
        _rewrite_member(target, capture.CHECKSUMS, b"0" + ledger[1:])
    elif mutation == "fifo":
        os.chmod(target, 0o700)
        victim = target / capture.RESOURCE_STDOUT
        victim.unlink()
        os.mkfifo(victim, 0o444)
        os.chmod(target, 0o555)
    elif mutation == "hardlink":
        os.chmod(target, 0o700)
        duplicate = target / "duplicate-hardlink"
        os.link(target / module_name, duplicate, follow_symlinks=False)
        os.chmod(target, 0o555)
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-entry",
        "member-mode",
        "target-mode",
        "marker-duplicate-key",
        "marker-nonfinite",
        "marker-hash",
        "marker-target-inode",
        "receipt-schema",
        "uppercase-sha",
        "ledger",
        "fifo",
        "hardlink",
    ],
)
def test_strict_verifier_rejects_malformed_mode_hash_ledger_and_entry_mutations(
    tmp_path: Path, mutation: str
):
    module, target, _ = _publish(tmp_path)
    try:
        _mutate_completed_capture(target, module.name, mutation)
        with pytest.raises((OSError, RuntimeError)):
            verifier.verify_capture_path(target)
    finally:
        _make_removable(target)


def test_failure_cleanup_never_uses_path_chmod_or_follows_an_injected_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, cuobjdump = _inputs(tmp_path)
    target = tmp_path / "capture"
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    os.chmod(victim, 0o640)
    real_claim = capture._claim_target

    def claim_and_inject(parent_fd: int, target_name: str):
        target_fd, metadata = real_claim(parent_fd, target_name)
        os.symlink(victim, "attacker-link", dir_fd=target_fd)
        return target_fd, metadata

    def path_chmod_is_forbidden(*_args, **_kwargs):
        raise AssertionError("capture cleanup must not chmod through a pathname")

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("simulated early capture failure")

    monkeypatch.setattr(capture, "_claim_target", claim_and_inject)
    monkeypatch.setattr(capture, "_run_cuobjdump", fail_capture)
    monkeypatch.setattr(Path, "chmod", path_chmod_is_forbidden)
    with pytest.raises(RuntimeError, match="simulated early capture failure"):
        capture.capture_artifacts(module, target, cuobjdump=cuobjdump)
    assert victim.read_bytes() == b"victim"
    assert _mode(victim) == 0o640
    assert (target / "attacker-link").is_symlink()
    _assert_incomplete(target)


def test_v3_contract_names_machine_readable_security_and_state_guarantees():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    serialized = json.dumps(contract, sort_keys=True)
    for required in (
        capture.CAPTURE_SCHEMA,
        capture.PUBLICATION_SCHEMA,
        "O_EXCL",
        "O_NOFOLLOW",
        "dirfd",
        "incomplete",
        "indeterminate",
        "no_clobber",
        "verifier",
    ):
        assert required in serialized


def test_prepare_capture_is_default_off_and_immediately_precedes_resource_gate():
    text = PREPARE.read_text(encoding="utf-8")
    variable = "FP4_LOCALCTA_PRE_RESOURCE_CAPTURE_DIR"
    capture_command = (
        'python "$(dirname "$0")/capture_localcta_pre_resource_artifacts.py"'
    )
    checker_command = 'python "$(dirname "$0")/check_localcta_runtime_resources.py"'

    assert f"if [[ -v {variable} ]]" in text
    assert f'if [[ -z "${{{variable}}}" ]]' in text
    assert f'--output-dir "${{{variable}}}"' in text
    assert f"{variable}:-" not in text
    capture_offset = text.index(capture_command)
    checker_offset = text.index(checker_command, capture_offset)
    restriction_offset = text.index(f"if [[ -v {variable} ]]")
    build_offset = text.index('arch_probe_dir="$(mktemp -d)"')
    assert restriction_offset < build_offset < capture_offset < checker_offset
    between = text[capture_offset:checker_offset]
    assert between.count("capture_localcta_pre_resource_artifacts.py") == 1
    assert "check_localcta_runtime_resources.py" not in between
