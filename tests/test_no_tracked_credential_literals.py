from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.credential_literal_scanner import scan_blob


def _tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"]
    )
    return [
        root / raw.decode("utf-8", "surrogateescape")
        for raw in output.split(b"\0")
        if raw
    ]


def find_tracked_credential_literals(root: Path) -> list[tuple[str, str]]:
    violations: set[tuple[str, str]] = set()
    for path in _tracked_files(root):
        try:
            data = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            continue
        relative = path.relative_to(root).as_posix()
        for category in scan_blob(relative, data):
            violations.add((category, relative))
    return sorted(violations)


@pytest.mark.parametrize(
    ("category", "path", "payload"),
    [
        (
            "aws-access-id",
            "binary.dat",
            b"\0prefix-" + b"AK" + b"IA" + b"A" * 16,
        ),
        (
            "aws-secret-assignment",
            "config.json",
            b'{"secretAccessKey":"' + b"S" * 40 + b'"}',
        ),
        (
            "aws-session-assignment",
            "config.yaml",
            b"sessionToken: " + b"T" * 96,
        ),
        (
            "api-key-assignment",
            "settings.py",
            b'config = {"api_key": "' + b"a" * 40 + b'"}',
        ),
        (
            "client-secret-assignment",
            "settings.json",
            b'{"clientSecret":"' + b"A!b#C$d%E^f&G*h~" * 2 + b'"}',
        ),
        (
            "provider-token-assignment",
            "settings.py",
            b'HF_TOKEN = decrypt("' + b"A!b#C$d%E^f&G*h~" * 2 + b'")',
        ),
        (
            "provider-token-assignment",
            "settings.py",
            b'HF_TOKEN = os.getenv("HF_TOKEN", "'
            + b"nonempty-fallback-value-123456"
            + b'")',
        ),
        (
            "aws-secret-assignment",
            "pod.yaml",
            b"- name: AWS_SECRET_ACCESS_KEY\n  value: " + b"S" * 40,
        ),
        (
            "aws-session-assignment",
            "escaped.json",
            b'{"command":"export AWS_SESSION_TOKEN=\\\"'
            + b"T" * 96
            + b'\\\""}',
        ),
        (
            "wandb-cli-key",
            "launch.sh",
            b"wandb login " + b"a" * 40,
        ),
        (
            "api-key-flag",
            "launch.sh",
            b"tool --api-key=" + b"a" * 40,
        ),
        (
            "wandb-url-key",
            "launch.txt",
            b"https://api.wandb.ai/authorize?api_key=" + b"a" * 40,
        ),
        (
            "github-token",
            "nul.bin",
            b"\0" + b"gh" + b"p_" + b"A" * 36,
        ),
        (
            "gitlab-token",
            "git.txt",
            b"gl" + b"pat-" + b"A" * 24,
        ),
        (
            "bitbucket-token",
            "git.txt",
            b"AT" + b"BB" + b"A" * 24,
        ),
        (
            "huggingface-token",
            "models.txt",
            b"h" + b"f_" + b"A" * 32,
        ),
        (
            "openai-api-key",
            "api.txt",
            b"s" + b"k-" + b"A" * 32,
        ),
        (
            "private-key-block",
            "key.txt",
            b"-----BEGIN ENCRYPTED " + b"PRIVATE KEY-----",
        ),
        (
            "pgp-private-key-block",
            "key.txt",
            b"-----BEGIN PGP " + b"PRIVATE KEY BLOCK-----",
        ),
        (
            "embedded-auth-url",
            "remote.txt",
            b"https://oauth2:" + b"A" * 24 + b"@github.com/repo",
        ),
        (
            "token-auth-url",
            "remote.txt",
            b"https://" + b"gh" + b"p_" + b"A" * 36 + b"@github.com/repo",
        ),
        (
            "sensitive-credential-filename",
            ".env",
            b"MODE=test",
        ),
        (
            "sensitive-credential-filename",
            "home/.docker/config.json",
            b"{}",
        ),
        (
            "sensitive-credential-filename",
            ".npmrc",
            b"registry=https://registry.example.invalid",
        ),
        (
            "sensitive-xtrace",
            "launch.sh",
            b"set -x\nprintf '%s' \"$WANDB_API_KEY\"\n",
        ),
    ],
)
def test_positive_adversarial_fixtures(
    category: str,
    path: str,
    payload: bytes,
) -> None:
    assert category in scan_blob(path, payload)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("config.json", b'{"secretAccessKey":"$' + b'{AWS_SECRET_ACCESS_KEY}"}'),
        (
            "pod.yaml",
            b"- name: AWS_SESSION_TOKEN\n"
            b"  valueFrom:\n"
            b"    secretKeyRef:\n"
            b"      name: training-credentials\n",
        ),
        ("config.yaml", b'api_key: "{{ wandb_api_key }}"'),
        (
            "remote.txt",
            b"https://oauth2:$" + b"{GITHUB_TOKEN}@github.com/repo",
        ),
        ("launch.sh", b"tool --api-key $" + b"{WANDB_API_KEY}"),
        ("launch.sh", b"set -e\nprintf '%s' \"$WANDB_API_KEY\"\n"),
        ("settings.py", b'HF_TOKEN = os.getenv("HF_TOKEN")'),
        ("settings.py", b'HF_TOKEN = os.getenv("HF_TOKEN", "")'),
        ("settings.py", b'HF_TOKEN = os.getenv("HF_TOKEN", None)'),
        ("settings.py", b'HF_TOKEN = os.environ["HF_TOKEN"]'),
        ("settings.py", b'missing.append("HF_TOKEN")'),
        ("public.pem", b"-----BEGIN PUBLIC KEY-----"),
        (".env.example", b"WANDB_API_KEY=$" + b"{WANDB_API_KEY}"),
        (".npmrc.example", b"registry=https://registry.example.invalid"),
    ],
)
def test_placeholder_and_reference_negatives(path: str, payload: bytes) -> None:
    assert scan_blob(path, payload) == set()


def test_tracked_tree_contains_no_credential_literals() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = find_tracked_credential_literals(root)
    assert not violations, (
        "credential-shaped literals found; reporting categories and paths only: "
        f"{violations}"
    )


def test_generic_run_train_keeps_credentials_runtime_only_without_xtrace() -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "run_train.sh"
    script = script_path.read_bytes()
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert b"set -x" not in script
    assert b"set -ex" not in script
    assert b"must be injected" not in script
    for forbidden in (
        b"export WANDB_API_KEY=",
        b"export HF_TOKEN=",
        b"export AWS_ACCESS_KEY_ID=",
        b"export AWS_SECRET_ACCESS_KEY=",
        b"export AWS_SESSION_TOKEN=",
    ):
        assert forbidden not in script
    assert scan_blob("run_train.sh", script) == set()
