# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
"""A module with utilities to help loading pytorch distributed checkpoints with streaming.

This helps reduce the memory footprint of operations which analyse checkpoints.
"""

from typing import cast, IO, Union, Sequence, Optional, Generator, Any
import io
import os
from contextlib import contextmanager
from pathlib import Path
import gc
from collections import defaultdict

import tqdm

import torch
import torch.distributed.checkpoint as dcp
from torch import Tensor
from torch.distributed.checkpoint import utils as dcp_utils
from torch.distributed.checkpoint.filesystem import LoadItemType
from torch.distributed.checkpoint import planner_helpers


@contextmanager
def self_fs_create_stream(path, mode="rb"):
    """Utility copied from torch dcp to support stream_checkpoint_reader"""
    if not isinstance(path, Path):
        path = Path(path)
    with path.open(mode) as stream:
        yield cast(io.IOBase, stream)


def self_fs_concat_path(
    path: Union[str, os.PathLike], suffix: str
) -> Union[str, os.PathLike]:
    """Utility copied from torch dcp to support stream_checkpoint_reader"""
    if not isinstance(path, Path):
        path = Path(path)
    return path / suffix


def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    """
    Utility copied from torch dcp to support stream_checkpoint_reader

    Copied from torch/distributed/_shard/_utils.py
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


def get_metadata(path: Union[str, os.PathLike]) -> dcp.Metadata:
    """Utility to extract metadata from a distributed checkpoint.

    Initial code pattern copied from
    https://github.com/pytorch/pytorch/blob/034e951b0cfb02d7b55327cd482e58cf2695dca0/torch/distributed/checkpoint/state_dict_loader.py#L373
    """
    checkpoint_id = str(path)
    storage_reader = dcp.state_dict_loader.cast(
        dcp.StorageReader,
        dcp.state_dict_loader._storage_setup(None, checkpoint_id, reader=True),
    )
    metadata = storage_reader.read_metadata()
    return metadata


def stream_checkpoint_reader(
    path: Union[str, os.PathLike],
    batch_tensors: int = 1,
    tensors_to_load: Optional[list[str]] = None,
    progress: bool = False,
) -> Generator[tuple[str, torch.Tensor], Any, None]:
    """Iterate over the tensors of a distributed checkpoint with no other information than it's path

    Why iterate? Checkpoints get very large, and loading them fully in memory becomes rapidly impossible.
    This function lets us process large checkpoints on reasonably sized machines.

    This function also makes an allowance for requesting specific tensors by name, without knowing their size in
    advance. To optimize read speed when processing entire checkpoints, they can be batched.

    Args:
        path: Path to the directory containing a Pytorch Distributed Checkpoint
        batch_tensors: Number of tensors to be loaded at each step (tensors are still returned 1 by 1)
            but this setting affects the read speed.
        tensors_to_load: specify tensors to load by fqn. Use ``get_metadata(path)state_dict_metadata.keys()``
            to see the names of the tensors.

    """
    metadata = get_metadata(path)
    meta_tensor_dict = {}
    fqns = tensors_to_load or list(metadata.state_dict_metadata.keys())
    for i in range(0, len(fqns), batch_tensors):
        batch_fqns = fqns[i : i + batch_tensors]
        for fqn in batch_fqns:
            tensor_metadata = metadata.state_dict_metadata[fqn]
            if not isinstance(tensor_metadata, dcp.metadata.TensorStorageMetadata):
                continue
            meta_tensor_dict[fqn] = torch.empty(
                tensor_metadata.size, dtype=tensor_metadata.properties.dtype
            )
        _read_data(path, metadata, meta_tensor_dict, progress)
        for fqn in meta_tensor_dict:
            yield fqn, meta_tensor_dict[fqn]
        for fqn in batch_fqns:
            if fqn in meta_tensor_dict:
                del meta_tensor_dict[fqn]
        gc.collect()


def _read_data(
    path: Union[str, os.PathLike],
    metadata: dcp.Metadata,
    meta_tensor_dict: dict[str, torch.Tensor],
    progress: bool,
) -> None:
    """
    Read data for tensors in meta_tensor_dict from checkpoint files.
    Optimizes filesystem access by grouping and sorting reads by offset.

    Args:
        path: Path to checkpoint directory
        metadata: Checkpoint metadata containing storage information.
        meta_tensor_dict: Dictionary mapping FQNs to target tensors.
            Will be empty tensors that are filled in place.


    This function follows a very similar pattern to `dcp.load`'s FileSystemReader.read_data
    https://github.com/pytorch/pytorch/blob/034e951b0cfb02d7b55327cd482e58cf2695dca0/torch/distributed/checkpoint/filesystem.py#L866

    Checkpoint tensors are loaded 1 by 1 to avoid requiring a lot of memory.

    Comments starting with `lbt: ` indicate the changes that were made when copying the code.
    """

    # Create read items for each tensor in meta_tensor_dict
    all_read_items = []
    for fqn, target_tensor in meta_tensor_dict.items():
        if fqn not in metadata.state_dict_metadata:
            continue
        tensor_metadata = metadata.state_dict_metadata[fqn]
        if not isinstance(tensor_metadata, dcp.metadata.TensorStorageMetadata):
            continue

        read_items = planner_helpers._create_read_items(
            fqn, md=tensor_metadata, obj=target_tensor
        )
        all_read_items.extend(read_items)

    # Group read items by file and sort by offset for optimal filesystem access
    per_file = defaultdict(list)
    for read_item in all_read_items:
        item_md = metadata.storage_data[read_item.storage_index]
        relative_path = item_md.relative_path
        per_file[relative_path].append((read_item, item_md))
    progress_func = tqdm.tqdm
    if not progress:
        progress_func = lambda x, *_, **__: x  # noqa: E731
    # Process each file
    for relative_path, items in progress_func(
        per_file.items(), total=len(per_file), desc="Reading checkpoint files"
    ):
        # Sort items by file offset to optimize sequential reads
        items.sort(key=lambda x: x[1].offset)

        new_path = self_fs_concat_path(path, relative_path)
        with self_fs_create_stream(new_path, "rb") as stream:
            for read_item, item_md in items:
                # Create file view for this specific read
                file_slice = cast(
                    IO[bytes],
                    dcp_utils._create_file_view(stream, item_md.offset, item_md.length),
                )

                # lbt: Transform load stream can be none - in which case "transform_from" just matches "file_slice"
                # leaving the code here for reference.
                # transform_from = self.transforms.transform_load_stream(
                #     req,
                #     # This field wasn't present in older
                #     # implementations so provide a fallback.
                #     item_md.transform_descriptors or (),
                #     file_slice,
                # )
                transform_from = file_slice

                if read_item.type == LoadItemType.BYTE_IO:
                    # lbt: I am unsure when this branch is hit. It has not been a problem (so far).
                    # Code is left commented for reference.
                    # read_bytes = io.BytesIO(transform_from.read(-1))
                    # read_bytes.seek(0)
                    # planner.load_bytes(req, read_bytes)
                    raise NotImplementedError("BYTE_IO loading not implemented")
                else:  # TensorType
                    # Ensure we have a seekable stream for torch.load
                    if transform_from.seekable():
                        seekable = transform_from
                    else:
                        seekable = io.BytesIO(transform_from.read(-1))
                        seekable.seek(0)

                    # Load tensor from file
                    tensor = cast(
                        Tensor,
                        torch.load(
                            seekable,
                            map_location="cpu",
                            weights_only=True,
                        ),
                    )

                    # lbt: implementation to output the tensor is different to the reference
                    # to flatten out the calls inside the `planner` object that does not exist
                    # in our implementation.
                    # Narrow the loaded tensor to the required slice
                    tensor = narrow_tensor_by_index(
                        tensor, read_item.storage_offsets, read_item.lengths
                    )

                    # Find the target shard in the destination tensor
                    target_tensor = meta_tensor_dict[read_item.dest_index.fqn]
                    target_shard = dcp_utils.find_tensor_shard(
                        target_tensor, read_item.dest_index
                    )
                    target_shard = narrow_tensor_by_index(
                        target_shard, read_item.dest_offsets, read_item.lengths
                    )

                    # Verify sizes match before copying
                    assert target_shard.size() == tensor.size(), (
                        f"req {read_item.storage_index} mismatch sizes "
                        f"{target_shard.size()} vs {tensor.size()}"
                    )

                    # Copy data to target
                    target_shard.copy_(tensor)
