# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from pathlib import Path
import hashlib
import low_bits_training  # noqa: F401


def test_TT_SPECIAL_ENTRIES_STATE_DICT_ENTRIES_is_up_to_date():
    import torchtitan.components.checkpoint

    content = Path(torchtitan.components.checkpoint.__file__).read_bytes()
    current_hash = hashlib.sha256(content).hexdigest()

    # Hash of the file when this test was written
    valid_hashes = [
        "63bfe5b84ecdd1ed8b94d746320337383221a583bc62e182d9d6416f7cf048c2",
        "c199459578af36c3bca398d8b73c524bc473cc60194660cdb5cdaa88089914d0",
    ]

    assert current_hash in valid_hashes, (
        f"checkpoint.py has changed. "
        f"Review TT_SPECIAL_ENTRIES_STATE_DICT_ENTRIES and CheckpointManager_dcp_load_override, then update the valid hash to: {current_hash}"
    )
