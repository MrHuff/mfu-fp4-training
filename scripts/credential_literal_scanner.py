from __future__ import annotations

from pathlib import PurePosixPath
import re


Finding = tuple[str, str]


_DIRECT_SIGNATURES = {
    "aws-access-id": re.compile(
        rb"(?<![A-Z0-9])(?:A3T[A-Z0-9]|ABIA|ACCA|AGPA|AIDA|AIPA|AKIA|"
        rb"ANPA|ANVA|AROA|ASCA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
    ),
    "github-token": re.compile(
        rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        rb"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9])"
    ),
    "gitlab-token": re.compile(
        rb"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"
    ),
    "bitbucket-token": re.compile(
        rb"(?<![A-Za-z0-9])ATBB[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"
    ),
    "huggingface-token": re.compile(
        rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"
    ),
    "google-api-key": re.compile(
        rb"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}(?![A-Za-z0-9])"
    ),
    "slack-token": re.compile(
        rb"(?<![A-Za-z0-9])xox(?:a|b|p|r|s)-[A-Za-z0-9-]{20,}"
    ),
    "openai-api-key": re.compile(
        rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"
    ),
    "private-key-block": re.compile(
        rb"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"
    ),
    "pgp-private-key-block": re.compile(
        rb"-----BEGIN PGP " rb"PRIVATE KEY BLOCK-----"
    ),
    "putty-private-key": re.compile(
        rb"(?im)^PuTTY-User-Key-File-[0-9]+:"
    ),
}

_KEY_GROUPS = {
    "aws-secret-assignment": (
        rb"AWS_SECRET_ACCESS_KEY|aws_secret_access_key|"
        rb"secretAccessKey|secret_access_key"
    ),
    "aws-session-assignment": (
        rb"AWS_SESSION_TOKEN|aws_session_token|sessionToken|session_token"
    ),
    "api-key-assignment": (
        rb"WANDB_API_KEY|api_key|apiKey|wandbApiKey|wandb_api_key"
    ),
    "provider-token-assignment": (
        rb"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|"
        rb"GITLAB_TOKEN|BITBUCKET_TOKEN|OPENAI_API_KEY"
    ),
    "client-secret-assignment": rb"client_secret|clientSecret",
}
_KEY_TO_KIND = {
    key.lower(): kind
    for kind, alternatives in _KEY_GROUPS.items()
    for key in alternatives.split(b"|")
}
_KEY_ALTERNATIVES = b"|".join(_KEY_GROUPS.values())
_VALUE = (
    rb"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|"
    rb"(?:os\.)?environ\[[^\]\r\n]+\]|"
    rb"[A-Za-z_][A-Za-z0-9_.]*\([^)\r\n]*\)|[^\s,;}\]]+)"
)
_LINE_ASSIGNMENT = re.compile(
    rb"(?im)^\s*(?:export\s+)?[\"']?(?P<key>" + _KEY_ALTERNATIVES + rb")"
    rb"[\"']?\s*(?:=|:)\s*" + _VALUE,
    re.MULTILINE,
)
_MAPPING_ASSIGNMENT = re.compile(
    rb"(?i)(?:[{,]\s*)[\"'](?P<key>" + _KEY_ALTERNATIVES + rb")"
    rb"[\"']\s*:\s*" + _VALUE
)
_INLINE_EXPORT = re.compile(
    rb"(?i)\bexport\s+(?P<key>" + _KEY_ALTERNATIVES + rb")"
    rb"\s*=\s*" + _VALUE
)
_ASSIGNMENTS = (_LINE_ASSIGNMENT, _MAPPING_ASSIGNMENT, _INLINE_EXPORT)
_NAME_VALUE = re.compile(
    rb"(?is)[\"']?name[\"']?\s*:\s*[\"']?"
    rb"(?P<key>" + _KEY_ALTERNATIVES + rb")[\"']?"
    rb".{0,256}?[\"']?value[\"']?\s*:\s*"
    rb"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)
_WANDB_CLI = re.compile(
    rb"(?i)\bwandb\s+login(?:\s+--relogin)?(?:\s+--key)?\s+"
    rb"[\"']?([^\s\"',;}\]]+)"
)
_API_KEY_FLAG = re.compile(
    rb"(?i)(?:--wandb-api-key|--api-key)\s*(?:=|\s)\s*"
    rb"[\"']?([^\s\"',;}\]]+)"
)
_WANDB_URL_KEY = re.compile(
    rb"(?i)https?://[^\s\"']*(?:api\.wandb\.ai|wandb\.ai)[^\s\"']*"
    rb"(?:api_key|key)=([^&\s\"']+)"
)
_URI_USERINFO = re.compile(
    rb"(?i)\b(?:https?|ssh|git|ftp)://([^/\s:@]+):([^/\s@]+)@"
)
_URI_TOKEN_USERINFO = re.compile(
    rb"(?i)\b(?:https?|ssh|git|ftp)://([^/\s:@]+)@"
)
_XTRACE = re.compile(
    rb"(?im)^\s*(?:set\s+-[A-Za-z]*x[A-Za-z]*|set\s+-o\s+xtrace)\b"
)
_SENSITIVE_KEY_REFERENCE = re.compile(
    rb"(?i)\b(?:" + _KEY_ALTERNATIVES + rb")\b"
)

_EXACT_PLACEHOLDERS = {
    b"",
    b"null",
    b"none",
    b"redacted",
    b"<redacted>",
    b"replace_me",
    b"replace-me",
    b"changeme",
    b"example",
    b"example_key",
    b"example-key",
    b"dummy",
    b"fake",
    b"test",
    b"username",
    b"user",
    b"password",
    b"pass",
    b"token",
    b"key",
    b"your_key_here",
    b"your-key-here",
    b"xxxxxxxxxxxxxxxx",
    b"0000000000000000",
}
_REFERENCE = re.compile(
    rb"(?i)^(?:"
    rb"\$\{\{[^{}\r\n]+\}\}|"
    rb"\$\{?[A-Z_][A-Z0-9_]*\}?|"
    rb"\{\{[^{}\r\n]+\}\}|"
    rb"<[A-Z_][A-Z0-9_.:-]*>|"
    rb"%[A-Z_][A-Z0-9_]*%|"
    rb"(?:secret|credential)(?:Ref|_ref)?[.:/].+"
    rb")$"
)
_ENV_REFERENCE = re.compile(
    rb"(?i)^(?:"
    rb"(?:os\.)?getenv\(\s*[\"'][A-Z_][A-Z0-9_]*[\"']"
    rb"(?:\s*,\s*(?:[\"'][\"']|None))?\s*\)|"
    rb"(?:os\.)?environ\.get\(\s*[\"'][A-Z_][A-Z0-9_]*[\"']"
    rb"(?:\s*,\s*(?:[\"'][\"']|None))?\s*\)|"
    rb"(?:os\.)?environ\[\s*[\"'][A-Z_][A-Z0-9_]*[\"']\s*\]"
    rb")$"
)
_HEX_40 = re.compile(rb"^[0-9a-fA-F]{40}$")

_SENSITIVE_BASENAMES = {
    ".env",
    ".netrc",
    "_netrc",
    "credentials",
    "credentials.json",
    "application_default_credentials.json",
    "service-account.json",
    "service_account.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    "kubeconfig",
}
_SENSITIVE_SUFFIXES = {".p12", ".pfx"}


def _normalize_for_context(data: bytes) -> bytes:
    return (
        data.replace(b"\\/", b"/")
        .replace(b'\\"', b'"')
        .replace(b"\\'", b"'")
        .replace(b"\\r\\n", b"\n")
        .replace(b"\\n", b"\n")
    )


def _is_reference_or_placeholder(value: bytes) -> bool:
    stripped = value.strip().strip(b"\"'")
    lowered = stripped.lower()
    return (
        lowered in _EXACT_PLACEHOLDERS
        or _REFERENCE.fullmatch(stripped) is not None
        or _ENV_REFERENCE.fullmatch(stripped) is not None
    )


def _value_is_credential_like(kind: str, value: bytes) -> bool:
    value = value.strip().strip(b"\"'")
    if _is_reference_or_placeholder(value):
        return False
    if any(pattern.search(value) for pattern in _DIRECT_SIGNATURES.values()):
        return True
    if kind == "api-key-assignment":
        return _HEX_40.fullmatch(value) is not None or len(value) >= 32
    minimum = {
        "aws-secret-assignment": 32,
        "aws-session-assignment": 32,
        "provider-token-assignment": 20,
        "client-secret-assignment": 20,
        "wandb-cli-key": 32,
        "wandb-url-key": 32,
    }.get(kind, 32)
    return len(value) >= minimum


def _sensitive_filename(path: str) -> bool:
    parsed = PurePosixPath(path.lower())
    return (
        parsed.name in _SENSITIVE_BASENAMES
        or parsed.suffix in _SENSITIVE_SUFFIXES
        or tuple(parsed.parts[-2:]) in {
            (".aws", "credentials"),
            (".docker", "config.json"),
        }
    )


def scan_blob(path: str, data: bytes) -> set[str]:
    """Return credential finding categories without returning matched values."""

    findings: set[str] = set()
    if _sensitive_filename(path):
        findings.add("sensitive-credential-filename")

    # Provider signatures and private-key headers are scanned over every raw
    # byte, including binary/NUL-containing members.
    for kind, pattern in _DIRECT_SIGNATURES.items():
        if pattern.search(data):
            findings.add(kind)

    # Contextual parsing is text-only; direct signatures above are never
    # bypassed by this binary guard.
    if b"\0" in data[:8192]:
        return findings

    normalized = _normalize_for_context(data)
    for assignment in _ASSIGNMENTS:
        for match in assignment.finditer(normalized):
            key = match.group("key").lower()
            kind = _KEY_TO_KIND[key]
            if _value_is_credential_like(kind, match.group("value")):
                findings.add(kind)
    for match in _NAME_VALUE.finditer(normalized):
        key = match.group("key").lower()
        kind = _KEY_TO_KIND[key]
        if _value_is_credential_like(kind, match.group("value")):
            findings.add(kind)

    for match in _WANDB_CLI.finditer(normalized):
        if _value_is_credential_like("wandb-cli-key", match.group(1)):
            findings.add("wandb-cli-key")
    for match in _API_KEY_FLAG.finditer(normalized):
        if _value_is_credential_like("api-key-assignment", match.group(1)):
            findings.add("api-key-flag")
    for match in _WANDB_URL_KEY.finditer(normalized):
        if _value_is_credential_like("wandb-url-key", match.group(1)):
            findings.add("wandb-url-key")

    for match in _URI_USERINFO.finditer(normalized):
        password = match.group(2)
        if not _is_reference_or_placeholder(password):
            findings.add("embedded-auth-url")
    for match in _URI_TOKEN_USERINFO.finditer(normalized):
        token = match.group(1)
        if (
            not _is_reference_or_placeholder(token)
            and _value_is_credential_like("provider-token-assignment", token)
        ):
            findings.add("token-auth-url")

    shell_code = b"\n".join(
        line for line in normalized.splitlines() if not line.lstrip().startswith(b"#")
    )
    if (
        path.lower().endswith((".sh", ".bash"))
        and _XTRACE.search(shell_code)
        and _SENSITIVE_KEY_REFERENCE.search(shell_code)
    ):
        findings.add("sensitive-xtrace")

    return findings
