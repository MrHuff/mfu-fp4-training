#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import importlib.util
import sys
import itertools


def load_module_from_file(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


check_codeowners = load_module_from_file(
    "check_codeowners", ".github/scripts/check_codeowners.py"
)


def test__is_experiments_only_changed_files__positive_result():
    paths = [
        "low_bits_training/experiments/mxfp8.py",
        "tests/cpu/experiments/test_mxfp8.py",
    ]
    assert check_codeowners.is_experiments_only_changed_files(paths)


def test__is_experiments_only_changed_files__negative_result():
    paths = [
        "low_bits_training/init.py",
        "low_bits_training/experiments/mxfp8.py",
        "tests/cpu/experiments/test_mxfp8.py",
    ]
    assert not check_codeowners.is_experiments_only_changed_files(paths)


def test__parse_codeowners__proper_parsing():
    r = check_codeowners.parse_codeowners("CODEOWNERS")
    assert isinstance(r, dict)
    assert len(r) >= 2
    users = set(itertools.chain(*r.values()))
    assert "balancap" in users


def test__check_github_owners__positive_result():
    paths = [
        "low_bits_training/experiments/mxfp8/mxfp8.py",
        "tests/cpu/experiments/mxfp8/test_mxfp8.py",
    ]
    username = "balancap"
    r = check_codeowners.check_github_owners(paths, username, "CODEOWNERS")
    assert r


def test__check_github_owners__wrong_paths():
    paths = [
        "low_bits_training/init.py",
        "tests/cpu/experiments/test_mxfp8.py",
    ]
    username = "balancap"
    r = check_codeowners.check_github_owners(paths, username, "CODEOWNERS")
    assert not r


def test__check_github_owners__wrong_username():
    paths = [
        "low_bits_training/experiments/mxfp8.py",
        "tests/cpu/experiments/test_mxfp8.py",
    ]
    username = "user"
    r = check_codeowners.check_github_owners(paths, username, "CODEOWNERS")
    assert not r
