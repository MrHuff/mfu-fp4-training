#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path
from typing import Dict, Any, Literal
import json


from datasets import load_dataset, __version__ as datasets_version
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.distributed import split_dataset_by_node

from torchtitan.config import JobConfig
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets.text_datasets import (
    HuggingFaceTextDataset,
    _load_c4_dataset,
    DatasetConfig,
    _process_c4_text,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger

Datasets = Dataset | IterableDataset

# TorchTitan base directory for test datasets.
tt_basedir = Path(__file__).resolve().parents[2] / "torchtitan_submodule"

CLUSTER_CACHE_DIR = "/opt/mfu/EXTERNAL_PATH"
_DATALOADER_KWARGS = {
    "num_workers",
    "pin_memory",
    "prefetch_factor",
}


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _dataset_loader_kwargs(load_dataset_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in load_dataset_kwargs.items()
        if key not in _DATALOADER_KWARGS
    }


def _process_text(sample: Dict[str, Any]) -> str:
    """Process slimpajama dataset sample text."""
    return sample["text"]


def load_dataset_with_custom_cluster_path(
    path: str,
    name: str,
    split: str,
    cache_dir=None,
    **load_dataset_kwargs,
) -> Datasets:
    """On the cluster datasets may be downloaded with the huggingface CLI.

    There is no easy way for `load_dataset` to recognize such datasets as
    caches of the official datasets. So we provide a custom path to the dataset
    that was downloaded with the CLI."""
    # TODO: Make this more generic, so that either the download script can register
    # cached datasets by using a file in the cache path registering the mapping
    # between the dataset name and the path to the dataset. Or simply do the string
    # surgery to figure out possible cache paths for the dataset if the dataset was
    # downloaded with HF CLI.
    if cache_dir is None:
        cache_dir = CLUSTER_CACHE_DIR
    shuffle_kwargs = load_dataset_kwargs.pop("shuffle_kwargs", {})
    defaults = {
        "streaming": True,
        "verification_mode": "no_checks",
        "cache_dir": cache_dir,
    }
    kwargs = {**defaults, **load_dataset_kwargs}
    if path.startswith("s3://"):
        arrow_glob = (
            path
            if path.endswith(".arrow")
            else f"{path.rstrip('/')}/opt/mfu/EXTERNAL_PATH"
        )
        kwargs.pop("cache_dir", None)
        logger.info(
            "Loading S3 Arrow dataset %s with split %s and arguments: %s",
            arrow_glob,
            split,
            kwargs,
        )
        ds = load_dataset(
            "arrow",
            data_files={split: arrow_glob},
            split=split,
            **kwargs,
        )
        if shuffle_kwargs:
            ds = ds.shuffle(**shuffle_kwargs)
        return ds

    # Location where the dataset has been downloaded with the HF CLI
    cluster_path = {
        "HuggingFaceFW/fineweb-edu": f"{cache_dir}datasets--HuggingFaceFW--fineweb-edu/snapshots/4863ab07d7520451e6f73e2912ad8bfee7d97c11/",
        "cerebras/SlimPajama-627B": f"{cache_dir}cerebras___slim_pajama-627_b/default/0.0.0/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543",
        # You must target the snapshot directory for the dataset or it will not be loaded correctly (silently)
        "olmo-mix-1124-600b-dclm": f"{cache_dir}datasets--allenai--olmo-mix-1124/snapshots/8162bd79c6dc4fea470506531a8d791badc06b4b/",
        "shuffled-olmo-mix-1124": f"{cache_dir}shuffled-olmo-mix-1124/snapshots/dummy/",
    }
    if path == "olmo-mix-1124-600b-dclm":
        assert (
            datasets_version >= "4.0.1"
        ), "Need a datasets version >=4.0.1 for olmo-mix-1124: https://huggingface.co/datasets/allenai/olmo-mix-1124/discussions/14#687ea21ea0f98cbe84aa1bfc"

    dataset_to_load = cluster_path.get(path, path)
    if dataset_to_load != path:
        logger.info(f"Loading dataset from cluster path: {dataset_to_load}")
    if not Path(cache_dir).exists():
        logger.warning(f"Cluster cache directory {cache_dir} does not exist.")
        cache_dir = None
    if not Path(dataset_to_load).exists():
        logger.warning(
            f"Dataset not found at cluster path: {dataset_to_load} resorting to loading from HF: {path}"
        )
        dataset_to_load = path
    defaults["cache_dir"] = cache_dir
    kwargs = {**defaults, **load_dataset_kwargs}
    logger.info(
        f"Loading dataset {dataset_to_load} with name {name} and split {split} with kwargs: {kwargs}"
    )
    ds = load_dataset(
        dataset_to_load,
        name=name,
        split=split,
        **kwargs,
    )
    # shuffling doesn't save you your batches are still very similar.
    if shuffle_kwargs:
        ds = ds.shuffle(**shuffle_kwargs)
    logger.info(
        f"Loaded dataset {dataset_to_load} with name {name} and split {split}: {ds}"
    )
    return ds


# Custom GC dataset collection, working with `GCHuggingFaceTextDataset` class
# We need our own collection as the `loader` interface is taking `load_dataset_kwargs` additional arguments
GC_DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=lambda path, load_dataset_kwargs: _load_c4_dataset(
            path,
            split=(load_dataset_kwargs or {}).get("split", "train"),
        ),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path=str(tt_basedir / "tests/assets/c4_test"),
        loader=lambda path, load_dataset_kwargs: load_dataset(
            path, split="train", **load_dataset_kwargs
        ),
        sample_processor=_process_c4_text,
    ),
}
# Full slimpajama dataset: train and validation.
GC_DATASETS["slimpajama"] = DatasetConfig(
    path="cerebras/SlimPajama-627B",
    loader=lambda path, load_dataset_kwargs: load_dataset_with_custom_cluster_path(
        # Load dataset from cache with streaming=True as that gives much faster initialisation
        # and fast iteration - the drawback is that the order is different than previous runs
        # if you need backward compatibility for runs done before
        # a425ab777f8dce52dce89d5b7f86fd5787ef3014 (July 2025) use slimpajama-old
        path,
        name="default",
        split="train",
        **load_dataset_kwargs,
    ),
    sample_processor=_process_text,
)
GC_DATASETS["slimpajama-old"] = DatasetConfig(
    path="cerebras/SlimPajama-627B",
    loader=lambda path, load_dataset_kwargs: load_dataset(
        path,
        name="default",
        split="train",
        **load_dataset_kwargs,
    ),
    sample_processor=_process_text,
)
GC_DATASETS["slimpajama_val"] = DatasetConfig(
    path="cerebras/SlimPajama-627B",
    loader=lambda path, load_dataset_kwargs: load_dataset(
        path, name="default", split="validation", **load_dataset_kwargs
    ),
    sample_processor=_process_text,
)
# 6B small slimpajama, useful for testing.
GC_DATASETS["slimpajama-6b"] = DatasetConfig(
    path="DKYoon/SlimPajama-6B",
    loader=lambda path, load_dataset_kwargs: load_dataset(
        path, name="default", split="train", **load_dataset_kwargs
    ),
    sample_processor=_process_text,
)

# Fineweb-edu dataset, a large dataset for training models that smash MMLU benchmarks.
GC_DATASETS["fineweb-edu"] = DatasetConfig(
    path="HuggingFaceFW/fineweb-edu",
    loader=lambda path, load_dataset_kwargs: load_dataset_with_custom_cluster_path(
        path,
        name="default",
        split="train",
        **load_dataset_kwargs,
    ),
    sample_processor=_process_text,
)
GC_DATASETS["olmo-mix-1t"] = DatasetConfig(
    path="olmo-mix-1124-600b-dclm",
    loader=lambda path, load_dataset_kwargs: load_dataset_with_custom_cluster_path(
        path,
        name="default",
        split="train",
        **load_dataset_kwargs,
    ),
    sample_processor=_process_text,
)
GC_DATASETS["olmo-1t-shuffled"] = DatasetConfig(
    path="shuffled-olmo-mix-1124",
    loader=lambda path, load_dataset_kwargs: load_dataset_with_custom_cluster_path(
        path,
        name="default",
        split="train",
        **load_dataset_kwargs,
    ),
    sample_processor=_process_text,
)
# Production 8B/1.2B Dolma default. The cluster-staged shuffled OLMo mix is
# our Dolma-family training corpus, so expose it under the job-facing Dolma name.
GC_DATASETS["dolma"] = GC_DATASETS["olmo-1t-shuffled"]
GC_DATASETS["dolma-shuffled"] = GC_DATASETS["olmo-1t-shuffled"]

GC_DATASETS["fineweb-edu-10b"] = DatasetConfig(
    path="HuggingFaceFW/fineweb-edu",
    loader=lambda path, load_dataset_kwargs: load_dataset_with_custom_cluster_path(
        path, name="sample-10BT", split="train", **load_dataset_kwargs
    ),
    sample_processor=_process_text,
)

GC_DATASETS["wikipedia"] = DatasetConfig(
    path="wikipedia",
    loader=lambda path, load_dataset_kwargs: load_dataset(
        path, name="20220301.en", split="train", **load_dataset_kwargs
    ),
    sample_processor=_process_text,
)


def distribute_dataset_across_nodes(
    ds: Datasets,
    dp_rank: int,
    dp_world_size: int,
    dataset_node_distribution: Literal["shard", "hf-flaky-splitting"],
) -> Datasets:
    """
    Distribute a dataset across multiple nodes using the specified splitting method.
    Any new code is encouraged to use `shard` as the distribution method, as it works
    similarly for streamed and map style datasets.

    The bit of code below was hell to figure out.

    HF ``datasets.split_dataset_by_node``  handles streamed and map style datasets very differently. Leading to
    large differences in training behavior.
    Streaming is the preferred way to load large datasets, but data in a global batch will come from a
    contiguous part of the dataset, when using an Iterable dataset with split_dataset_by_node.
    Instead we try to use the shard() method first which will implicitly
    shuffle the data by mixing data from different files between replicas.
    We fall back to split_dataset_by_node if shard() fails (e.g. for streamed datasets from the web,
    and small local datasets that are not sharded).

    Why don't we just call shuffle on the dataset? Because the approximate shuffle provided by HF datasets
    is not sufficient to mix data from different files in the dataset inside a single global batch (even with
    very large buffers), so global batches will have very different data distributions leading to training instability.

    The only proper way to shuffle is to use a Map style dataset (not streamed) and call shuffle.
    The will rewrite the entire dataset on disk in a shuffled order. Do this before training and then
    use shard() during training.

    Another issue with split_dataset_by_node is that to restart you need to roll the iterator forward, and that
    iterator needs to load all the files that were skipped. The interleaved split from split_dataset_by_node
    increases the amount of data that needs to be read to roll the iterator forward from N_tokens to
    N_tokens * N_replicas.
    """
    if dataset_node_distribution == "shard":
        try:
            data = ds.shard(num_shards=dp_world_size, index=dp_rank, contiguous=True)
        except Exception as e:
            logger.error(
                f"shard() failed with {type(e).__name__}: {e}, pass --training.dataset_node_distribution=hf-flaky-splitting"
                " to use HuggingFace datasets' default approach instead."
            )
            raise
    elif dataset_node_distribution == "hf-flaky-splitting":
        logger.warning(
            "Using HuggingFace split_dataset_by_node to shard the dataset. "
            "This is not recommended, use --training.dataset_node_distribution=shard."
            " You risk inconsistencies in training behavior between streaming=True and False, and restarts"
            " may be slow and flaky."
        )
        data = split_dataset_by_node(ds, dp_rank, dp_world_size)
    else:
        raise ValueError(
            f"Unknown dataset_node_distribution: {dataset_node_distribution}"
        )
    return data


class GCHuggingFaceTextDataset(HuggingFaceTextDataset):
    def __init__(
        self,
        job_config: JobConfig,
        tokenizer: BaseTokenizer,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool = False,
    ):
        dataset_name = job_config.training.dataset
        dataset_path = job_config.training.dataset_path
        load_dataset_kwargs = json.loads(job_config.training.load_dataset_kwargs)
        load_dataset_kwargs = _dataset_loader_kwargs(load_dataset_kwargs)
        if dataset_name not in GC_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not found in low-bits-training datasets. "
                f"Support datasets are: {list(GC_DATASETS.keys())}"
            )
        dataset_config = GC_DATASETS[dataset_name]
        dataset_path = dataset_path or dataset_config.path
        # Main change compared to TorchTitan: add custom `load_dataset_kwargs` from Job config.
        logger.info(f"Using dataset {dataset_path} with arguments: {load_dataset_kwargs}")
        ds = dataset_config.loader(dataset_path, load_dataset_kwargs)
        self._data = distribute_dataset_across_nodes(
            ds, dp_rank, dp_world_size, job_config.training.dataset_node_distribution
        )
        self.dataset_name = dataset_name

        self._tokenizer = tokenizer
        self.seq_len = job_config.training.seq_len
        self.infinite = infinite
        self._text_processor = dataset_config.sample_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def state_dict(self) -> dict[str, Any]:
        """Return a stable snapshot of the mutable token buffer.

        TorchData retains dataset state for incremental worker snapshots and
        sends deltas through a multiprocessing queue.  Returning the live list
        lets the next fetch mutate a supposedly captured snapshot before the
        queue feeder has serialized it.  That produces timing-dependent,
        rank-local resume streams.  Copying the buffer makes the snapshot
        immutable from the dataset's point of view.
        """
        state = super().state_dict()
        state["token_buffer"] = list(state["token_buffer"])
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Do not let resumed iteration mutate the checkpoint object retained by
        # the checkpointer or a caller-side replay oracle.
        stable_state = dict(state_dict)
        stable_state["token_buffer"] = list(state_dict["token_buffer"])
        super().load_state_dict(stable_state)


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    batch_size = job_config.training.local_batch_size
    load_dataset_kwargs = json.loads(job_config.training.load_dataset_kwargs)
    num_workers = int(load_dataset_kwargs.get("num_workers", 0))
    prefetch_factor = int(load_dataset_kwargs.get("prefetch_factor", 4))
    pin_memory = _as_bool(load_dataset_kwargs.get("pin_memory"), default=False)
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )
    logger.info(
        "Hugging Face dataloader with batch size %s, %s workers, "
        "prefetch_factor=%s, pin_memory=%s.",
        batch_size,
        num_workers,
        prefetch_factor,
        pin_memory,
    )

    hf_ds = GCHuggingFaceTextDataset(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
