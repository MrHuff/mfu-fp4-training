#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""
OLMES Results Parser and Visualizer

This script processes OLMES evaluation results from model checkpoint directories.
It extracts metrics from each checkpoint step and allows for visualization of how
metrics evolve over the course of training.

The script expects OLMES results to be located at:
    step-X/transformers-checkpoint/evaluations/olmes/


Example notebook usage:

    import parse_olmes
    parse_olmes.process_checkpoint_folder("/opt/mfu/EXTERNAL_PATH")

Example command line usage:

    # Basic usage - process a checkpoint directory
    python parse_olmes.py --checkpoint-dir /path/to/checkpoints --output-dir results

    # Output flattened data for external plotting
    python parse_olmes.py --checkpoint-dir /path/to/checkpoints --flatten

"""

from typing import Literal
import json
import pandas as pd
import sys
import os
import re
from pathlib import Path
import argparse
import logging


from ..utils import find_config, load_config, wandb_eval_init


CheckpointTypes = Literal["", "ema_"]  # Type for checkpoint type, can be empty or 'ema_'

# Configure the logger
logger = logging.getLogger(__name__)


def parse_results(content: str) -> pd.DataFrame | None:
    """Parse OLMES evaluation results from a JSON Lines (jsonl) format string.

    This function processes the content of an OLMES results file, which contains
    one JSON object per line. Each line represents a task result with metrics.

    Args:
        content (str): String containing newline-separated JSON objects (jsonl format),
                       where each object represents results for a single evaluation task.

    Returns:
        pandas.DataFrame or None: DataFrame containing parsed task data with columns for
                                 task_name, processing_time, and all metrics found in the input.
                                 Returns None if no valid task data is found.

    Notes:
        - The function expects each JSON object to have at least 'task_name' and 'metrics' fields
        - Numeric values are rounded for readability (processing_time to 2 decimal places,
          metrics to 4 decimal places)
        - The returned DataFrame will have task_name and processing_time as the first columns
    """
    # Initialize a list to store all task data
    all_tasks_data = []

    # Parse each line in the content
    for line in content.strip().split("\n"):
        if not line.strip():
            continue

        try:
            data = json.loads(line)

            # Check if this appears to be a valid task result
            if "task_name" not in data or "metrics" not in data:
                continue

            # Extract required fields
            task_name = data["task_name"]
            processing_time = data.get("processing_time", 0)
            metrics = data["metrics"]

            # Create a dictionary for this task
            task_data = {"task_name": task_name, "processing_time": processing_time}

            # Add metrics to the dictionary
            for metric_name, metric_value in metrics.items():
                task_data[metric_name] = metric_value

            # Append to our list
            all_tasks_data.append(task_data)

        except json.JSONDecodeError:
            continue  # Skip lines that aren't valid JSON

    if not all_tasks_data:
        logger.warning("No valid task data found. Please check the input format.")
        return None

    # Create a DataFrame
    df = pd.DataFrame(all_tasks_data)

    # Round numeric columns for readability
    for col in df.columns:
        if col == "task_name":  # Skip non-numeric columns
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col == "processing_time":
                df[col] = df[col].round(2)  # Less precision for time
            else:
                df[col] = df[col].round(4)  # More precision for metrics
        except Exception as e:
            logger.error(f"Error rounding column {col}: {e}")

    # Reorder columns to put task_name and processing_time first
    cols = ["task_name", "processing_time"] + [
        col for col in df.columns if col not in ["task_name", "processing_time"]
    ]
    df = df[cols]

    return df


def extract_step_number(checkpoint_path: str | Path) -> int | None:
    """Extract step number from a checkpoint path"""
    match = re.search(r"step-(\d+)", Path(checkpoint_path).name)
    if match:
        return int(match.group(1))
    return None


def find_olmes_results(checkpoint_dir: str | Path) -> list[Path]:
    """Find OLMES results files in the checkpoint directory"""
    # Target path structure: step-X/transformers-checkpoint/evaluations/olmes/
    target_path = (
        Path(checkpoint_dir) / "transformers-checkpoint" / "evaluations" / "olmes"
    )

    # Look specifically for metrics-all.jsonl file
    metrics_all_file = target_path / "metrics-all.jsonl"

    if metrics_all_file.exists():
        logger.info(f"Found metrics-all.jsonl file in {target_path}")
        return [metrics_all_file]

    # Check if the target directory exists
    if not target_path.exists() or not target_path.is_dir():
        logger.warning(f"Target directory not found: {target_path}")
        # Fall back to general pattern search
        metrics_all_files = list(Path(checkpoint_dir).glob("**/metrics-all.jsonl"))
        if metrics_all_files:
            return metrics_all_files

    logger.warning(f"No OLMES results files found in {checkpoint_dir}")
    return []


def extract_training_params(checkpoint_folder: str | Path) -> tuple[int, int]:
    """
    Extract global batch size and sequence length from slurm output files

    Returns:
        tuple: (global_batch_size, sequence_length)
    """
    # Look for slurm output files in the checkpoint directory
    job_log_files = list(Path(checkpoint_folder).glob("*.out"))

    # If no slurm files in the directory, try looking in the parent directory
    if not job_log_files:
        parent_dir = Path(checkpoint_folder).parent
        job_log_files = list(parent_dir.glob("*.out"))

    global_batch_size = None
    sequence_length = None

    # Read slurm logs to find the line with "global batch size <number> sequence length <number>"
    for job_log_file in job_log_files:
        try:
            with open(job_log_file, "r") as f:
                for line in f:
                    if "global batch size" in line and "sequence length" in line:
                        # Extract global batch size
                        batch_match = re.search(r"global batch size (\d+)", line)
                        if batch_match:
                            global_batch_size = int(batch_match.group(1))

                        # Extract sequence length
                        seq_match = re.search(r"sequence length (\d+)", line)
                        if seq_match:
                            sequence_length = int(seq_match.group(1))

                        if global_batch_size and sequence_length:
                            logger.info(
                                f"Found training parameters: global_batch_size="
                                f"{global_batch_size}, sequence_length={sequence_length}"
                            )
                            return global_batch_size, sequence_length
        except Exception as e:
            logger.error(f"Error reading log file {job_log_file}: {e}")

    # If parameters were not found, use default values
    if global_batch_size is None:
        logger.warning(
            "Could not find global batch size in slurm output files. Using default value of 0."
        )
        global_batch_size = 0

    if sequence_length is None:
        logger.warning(
            "Could not find sequence length in slurm output files. Using default value of 0."
        )
        sequence_length = 0

    return global_batch_size, sequence_length


def process_checkpoint_folder(
    checkpoint_folder: str | Path,
    output_dir: str | Path | None = None,
    upload_to_wandb: bool = False,
    checkpoint_type: CheckpointTypes = "",
) -> pd.DataFrame | None:
    """Process a folder containing checkpoints, extract metrics for each step"""
    # Find all checkpoint step directories
    checkpoint_dirs = []

    # First approach: look for directories with pattern step-*
    for step_dir in Path(checkpoint_folder).glob(f"{checkpoint_type}checkpoint/step-*"):
        if step_dir.is_dir():
            checkpoint_dirs.append(step_dir)

    if not checkpoint_dirs:
        logger.warning(f"No checkpoint step directories found in {checkpoint_folder}")
        return None

    logger.info(f"Found {len(checkpoint_dirs)} checkpoint directories")

    # Extract training parameters for token count calculation
    global_batch_size, sequence_length = extract_training_params(checkpoint_folder)

    # Process each checkpoint
    results_data = []
    for cp_dir in sorted(checkpoint_dirs):
        step_number = extract_step_number(cp_dir)
        if step_number is None:
            logger.info(
                f"Could not extract step number from {cp_dir}, using directory name as identifier"
            )
            # Use the directory name as an identifier if we can't extract a step number
            step_number = Path(cp_dir).name

        # Find OLMES results files
        olmes_files = find_olmes_results(cp_dir)

        # Prioritize metrics-all.jsonl if present
        if metrics_all_files := [f for f in olmes_files if f.name == "metrics-all.jsonl"]:
            olmes_files = metrics_all_files
            logger.info(f"Using metrics-all.jsonl for step {step_number}")

        for olmes_file in olmes_files:
            try:
                results_df = parse_results(olmes_file.read_text())

                if results_df is None:
                    logger.warning(f"No valid results found in {olmes_file}")
                    continue
                # Add step information to each row
                results_df["step"] = step_number
                results_df["checkpoint_path"] = str(cp_dir)
                results_df["olmes_file"] = str(olmes_file)

                # Calculate token count: step * global_batch_size * sequence_length
                # Make sure step is a number for this calculation
                numeric_step = step_number
                if isinstance(step_number, str):
                    try:
                        numeric_step = int(re.search(r"(\d+)", step_number).group(1))
                    except Exception as e:
                        logger.warning(f"Error parsing step number {step_number}: {e}")
                        numeric_step = 0  # Default if step can't be parsed

                results_df["token_count"] = (
                    numeric_step * global_batch_size * sequence_length
                )

                # Store the results
                results_data.append(results_df)

                logger.info(f"Processed {olmes_file} from step {step_number}")
            except Exception as e:
                logger.error(f"Error processing {olmes_file}: {e}")

    if not results_data:
        logger.warning("No valid results found in any checkpoint directory")
        return None

    # Combine all results into a single DataFrame
    combined_df = pd.concat(results_data, ignore_index=True)

    # Convert step to numeric if possible (for proper sorting)
    try:
        combined_df["step"] = pd.to_numeric(combined_df["step"])
    except Exception as e:
        logger.warning(f"Error converting step to numeric: {e}")

    # Sort by step
    combined_df = combined_df.sort_values("step")

    # Save combined results if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"olmes_metrics_{Path(checkpoint_folder).name}.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"OLMES metrics saved to {output_file}")

    if upload_to_wandb:
        log_headline_metrics_to_wandb(combined_df, checkpoint_folder, checkpoint_type)
    return combined_df


def log_headline_metrics_to_wandb(
    df: pd.DataFrame, checkpoint_folder: str | Path, checkpoint_type: CheckpointTypes = ""
) -> None:
    """Upload the OLMES metrics to WANDB"""
    config = load_config(
        find_config(checkpoint_folder), allow_upgrade=True, validate=False
    )
    exclude = [
        "task_name",
        "token_count (B)",
        "processing_time",
        "token_count",
        "checkpoint_path",
        "olmes_file",
        "step",
    ]
    run = wandb_eval_init(config, checkpoint_type + "olmes", delete_previous=True)
    run.define_metric("evaluation/step")  # needed to store steps out of order
    run.define_metric("evaluation/token_count")  # needed to store steps out of order
    run.define_metric("evaluation/*", step_metric="evaluation/token_count")
    # Test the ability to log out of order
    steps = df["step"].unique()
    sorted(steps)
    for step in steps:
        df_wandb = df[df["step"] == step]
        o = df_wandb.to_dict(orient="split", index=False)
        d = [{c: v for c, v in zip(o["columns"], r) if not pd.isna(v)} for r in o["data"]]
        data = {
            "evaluation/step": step,
            "evaluation/token_count": df_wandb["token_count"].iloc[0],
        }
        for row in d:
            data[f"evaluation/olmes/{row['task_name']}"] = {
                k: v for k, v in row.items() if k not in exclude
            }
        run.log(data)
    run.finish()


def flatten_metrics_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Flatten the metrics DataFrame to make it easier to plot
    Convert from:
    step | task_name | metric1 | metric2 | ... | token_count

    To:
    step | task_name | metric_name | metric_value | token_count
    """
    if df is None:
        return None

    # Identify metric columns (exclude non-metric columns)
    non_metric_cols = [
        "step",
        "task_name",
        "processing_time",
        "checkpoint_path",
        "olmes_file",
        "token_count",
    ]
    metric_cols = [col for col in df.columns if col not in non_metric_cols]

    # Create an empty list to store flattened data
    flattened_data = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        # Extract non-metric values
        step = row["step"]
        task_name = row["task_name"]
        checkpoint_path = row.get("checkpoint_path", "")
        token_count = row.get("token_count", 0)

        # Create one row for each metric
        for metric_name in metric_cols:
            metric_value = row[metric_name]
            flattened_data.append(
                {
                    "step": step,
                    "task_name": task_name,
                    "checkpoint_path": checkpoint_path,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "token_count": token_count,
                }
            )

    # Create a new DataFrame from the flattened data
    flattened_df = pd.DataFrame(flattened_data)

    return flattened_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process OLMES results from checkpoint directories"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", help="Specific metrics to plot (default: all)"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", help="Specific tasks to include (default: all)"
    )
    parser.add_argument(
        "--flatten", action="store_true", help="Output flattened data format"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--upload-to-wandb",
        action="store_true",
        help="Upload the OLMES metrics to WANDB",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="",
        choices=CheckpointTypes.__args__,
        help="Type of checkpoint to process (default: '', 'ema_')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Process checkpoint directory
    logger.info(f"Processing checkpoints in: {args.checkpoint_dir}")
    results_df = process_checkpoint_folder(
        args.checkpoint_dir,
        args.output_dir,
        args.upload_to_wandb,
        args.checkpoint_type,
    )

    if results_df is None:
        logger.error("No results found or processing failed")
        return

    # Optionally flatten the data
    if args.flatten and args.output_dir:
        flattened_df = flatten_metrics_df(results_df)
        flattened_output = os.path.join(args.output_dir, "flattened_metrics.csv")
        flattened_df.to_csv(flattened_output, index=False)
        logger.info(f"Flattened metrics saved to {flattened_output}")

        # Use flattened data for subsequent operations
        results_df = flattened_df

    # Display summary of results
    logger.info("\nSummary of metrics:")
    tasks = results_df["task_name"].unique()
    for task in tasks:
        logger.info(f"\nTask: {task}")
        task_df = results_df[results_df["task_name"] == task]

        if "metric_name" in task_df.columns:  # Flattened format
            metrics = task_df["metric_name"].unique()
            for metric in metrics:
                metric_values = task_df[task_df["metric_name"] == metric]["metric_value"]
                logger.info(
                    f"  {metric}: min={metric_values.min():.4f}, max={metric_values.max():.4f}, latest={metric_values.iloc[-1]:.4f}"
                )
        else:  # Original format
            metrics = [
                col
                for col in task_df.columns
                if col
                not in [
                    "task_name",
                    "processing_time",
                    "step",
                    "checkpoint_path",
                    "olmes_file",
                    "token_count",
                ]
            ]
            for metric in metrics:
                logger.info(
                    f"  {metric}: min={task_df[metric].min():.4f}, max={task_df[metric].max():.4f}, latest={task_df[metric].iloc[-1]:.4f}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
