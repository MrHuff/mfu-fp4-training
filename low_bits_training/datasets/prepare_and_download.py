#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import re
import subprocess
import time
import argparse

import wandb

from ..utils import run_process_with_realtime_output


def download_dataset_with_hf_cli(args):
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=f"dataset-download-{args.hf_dataset}",
    )
    finished = False
    max_retry = 100
    retries = 0
    success = False
    while not finished:
        try:
            process = subprocess.Popen(
                [
                    "huggingface-cli",
                    "download",
                    args.hf_dataset,
                    "--repo-type",
                    "dataset",
                    f"--cache-dir={args.cache_dir}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,  # Line buffered
            )

            # Run the process and capture output in real-time
            return_code = run_process_with_realtime_output(process)
            assert return_code == 0, f"Process failed with return code {return_code}"
            finished = True
            success = True
        except Exception as e:
            print(f"Error while downloading: {e}")
            time.sleep(60)
            retries += 1
            finished = retries >= max_retry

    print("Finished downloading starting to copy folder")
    # Define the paths and hosts
    # This will work but we should use an attribute from the dataset object
    cache_path_name = re.sub(
        "[/]", "_", re.sub("([A-Z])", r"_\g<1>", args.hf_dataset).lower()
    )
    source_path = f"{args.cache_dir}/{cache_path_name}"

    if success:
        print("Dataset downloaded to ", source_path)
    else:
        print("Failed to download dataset after retries")

    run.finish()


def get_download_parser():
    parser = argparse.ArgumentParser(description="Download and copy dataset")

    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="Hugging Face dataset to download",
    )
    parser.add_argument("--entity", type=str, default="graphcore", help="Wandb entity")

    parser.add_argument(
        "--project",
        type=str,
        default="low-bits-training-dataset-download",
        help="Wandb project",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/opt/mfu/EXTERNAL_PATH",
        help="Cache directory for datasets",
    )

    return parser


if __name__ == "__main__":
    download_dataset_with_hf_cli(get_download_parser().parse_args())
