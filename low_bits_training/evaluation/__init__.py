#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from typing import Literal, NamedTuple
from abc import ABC, abstractmethod
import argparse
import subprocess

import os
import pathlib
import json
import re

import wandb as wandb

from ..checkpoints import convert_checkpoint_to_transformers
from ..utils import (
    wandb_id_from_config as wandb_id_from_config,
    load_config,
    wandb_eval_init,
    run_process_with_realtime_output,
)
from . import environment_manager
from . import parse_olmes as parse_olmes


class SharedArgs(NamedTuple):
    model_config: str
    model_checkpoint: str
    output_dir: str


class EvaluationBase(ABC):
    """Base class for evaluation methods.

    Attributes:
        checkpoint_format: The format of the checkpoint that the evaluation method expects.
        default_output_dir: The default output directory for the evaluation results, relative
            to the checkpoint directory or the output directory specified by the arguments.
    """

    checkpoint_format: Literal["torchtitan", "transformers"]
    default_output_dir: str

    @classmethod
    @abstractmethod
    def run(cls, shared_args: SharedArgs, *, own_args: argparse.Namespace, **kwargs): ...

    @classmethod
    @abstractmethod
    def get_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser: ...


def get_shared_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a model according to one of the supported methods of benchmarks."
    )
    subparsers = parser.add_subparsers(dest="method")
    for method_name, method in EVALUATION_METHODS.items():
        subparser = subparsers.add_parser(method_name)
        method.get_parser(subparser)

    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="The model config to evaluate the model on.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="The model checkpoint to evaluate the model on.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="The output directory to save the results.",
    )

    return parser


class OlmesEval(EvaluationBase):
    checkpoint_format = "transformers"
    default_output_dir = "evaluations/olmes"

    @classmethod
    def run(
        cls,
        shared_args: SharedArgs,
        *,
        own_args: argparse.Namespace,
        unknown_args: list[str] = [],
        **kwargs,
    ):
        shared_args.output_dir = shared_args.output_dir or cls.default_output_dir
        transformers_checkpoint_dir = convert_checkpoint_to_transformers(
            shared_args.model_checkpoint,
            shared_args.model_config,
            converted_checkpoint_dir=own_args.transformers_checkpoint_dir,
        )
        # runs the command: olmes --model ${CKPT} --model-args '{"dtype": "bfloat16"}' \
        # --task core_9mcqa::olmes mmlu::olmes --output-dir ${CKPT}/evaluations/olmes
        # Use Popen to pipe output in real-time
        olmes_output_dir = f"{transformers_checkpoint_dir}/{shared_args.output_dir}"
        with environment_manager.EnvironmentManager(
            [
                "git+https://github.com/graphcore-research/olmes-fork.git@sanitise-names[gpu]",
                "blobfile",
            ]
        ) as manager:
            process = manager.run_in_env(
                subprocess.Popen,
                [
                    "olmes",
                    "--model",
                    str(transformers_checkpoint_dir),
                    "--model-args",
                    own_args.model_args,
                    "--task",
                    *own_args.task,
                    "--output-dir",
                    olmes_output_dir,
                    *unknown_args,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                cwd=str(transformers_checkpoint_dir),
            )

            # Run the process and capture output in real-time
            return_code = run_process_with_realtime_output(process)

        # Log the evaluation metrics to wandb if requested
        if own_args.wandb:
            cls._log_to_wandb(
                olmes_output_dir, shared_args.model_checkpoint, shared_args.model_config
            )

        return return_code

    @classmethod
    def _log_to_wandb(
        cls, olmes_output_dir: str, model_checkpoint: str, model_config: str
    ):
        """
        Log the metrics in the evaluations/olmes/metrics_all.jsonl file to the WANDB training run

        Args:
            olmes_output_dir: The directory containing the OLMES output
            model_checkpoint: The path to the model checkpoint
            model_config: The path to the model config
        """
        if not pathlib.Path(olmes_output_dir).exists():
            raise ValueError(f"OLMES output directory {olmes_output_dir} does not exist")
        # Get the step from model checkpoint
        step = re.search(r"step-(\d+)", str(model_checkpoint))
        if not step:
            raise ValueError(
                f"Failed to get step number from model checkpoint {model_checkpoint} - cannot log to wandb"
            )
        try:
            step = int(step.group(1))
        except Exception as e:
            raise ValueError(
                f"Failed to convert step number {step.group(1)} from model checkpoint "
                f"{model_checkpoint} to int - cannot log to wandb"
            ) from e
        job_config = load_config(model_config)
        raise NotImplementedError(
            "Online WANDB logging is not implemented yet - wait for all processes to finish and run notebooks/parse_olmes.py"
        )
        # TODO: Make `wandb_eval_init` safe to call from multiple processes which want to log to the same run.
        # there is a potential race condition on creating the evaluation run
        # in particular, you need to make sure that multiple processes don't create runs with the same name at the same time.
        run = wandb_eval_init(job_config, "olmes")
        # Define a step metric to visualise results in an order different from the upload order
        # https://docs.wandb.ai/guides/track/log/customize-logging-axes/
        run.define_metric("evaluation/step")
        run.define_metric("evaluation/*", step_metric="evaluation/step")
        # Load the metrics_all.jsonl each line has a task_name string and metrics dict which need to be logged
        metrics_to_log = {"evaluation/step": step}
        with open(f"{olmes_output_dir}/metrics_all.jsonl", "r") as f:
            for line in f.splitlines():
                data = json.loads(line)
                task_name = data["task_name"]
                metrics = data["metrics"]
                metrics_to_log[f"evaluation/olmes/{task_name}"] = metrics
        run.log(metrics_to_log)
        # TODO: log the file as an artifact
        run.finish()

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # add task , output dir and model args - defaults should match above
        parser.add_argument(
            "--task", type=str, nargs="+", default=["core_9mcqa::olmes", "mmlu::olmes"]
        )
        parser.add_argument("--model-args", type=str, default='{"dtype": "bfloat16"}')
        parser.add_argument("--transformers-checkpoint-dir", type=str, default=None)
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="Enable wandb logging at the end of the evaluation",
        )
        parser.description = "Run OLMES evaluation. Additional arguments are passed directly to the olmes command run `olmes --help` for more information"
        return parser


class EleutherEval(EvaluationBase):
    checkpoint_format = "transformers"
    default_output_dir = "evaluations/eleuther"

    @classmethod
    def run(cls, shared_args: SharedArgs, *, own_args: argparse.Namespace, **kwargs):
        shared_args.output_dir = shared_args.output_dir or cls.default_output_dir
        raise NotImplementedError("EleutherEval integration is not implemented yet")
        # TODO: apply our checkpoint conversion and then call lm_eval in a similar way
        # to what is done in OlmesEval
        # transformers_checkpoint = "/opt/mfu/EXTERNAL_PATH"
        # lm_eval --model hf --model_args pretrained=${transformers_checkpoint},dtype=bfloat16 --tasks mmlu --device cuda:0
        # --batch_size auto --num_fewshot 5 --cache_requests true --log_samples --output_path
        # ${transformers_checkpoint}/evaluations/eleuther/

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser


def _get_args_from_namespace(namespace: argparse.Namespace, keys: list[str]) -> dict:
    """
    Get a dictionary of arguments from a namespace.
    """
    return {k: v for k, v in vars(namespace).items() if k in keys}


class GenerateFromPrompt(EvaluationBase):
    checkpoint_format = "torchtitan"
    default_output_dir = "evaluations/generate"

    @classmethod
    def run(cls, shared_args: SharedArgs, *, own_args: argparse.Namespace, **kwargs):
        shared_args.output_dir = shared_args.output_dir or cls.default_output_dir
        from low_bits_training.generate.generate import Generator

        generator = Generator(
            config_path=shared_args.model_config,
            checkpoint_path=shared_args.model_checkpoint,
            output_dir=shared_args.output_dir,
            **_get_args_from_namespace(own_args, ["seed", "deterministic"]),
        )
        out = generator.generate(
            prompt=own_args.prompt,
            **_get_args_from_namespace(
                own_args, ["temperature", "max_new_tokens", "batch_size", "top_k", "seed"]
            ),
        )
        print(out)

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        try:
            from low_bits_training.generate.generate import add_generation_args

            return add_generation_args(parser)
        except ImportError:
            return parser


class SimpleEval(EvaluationBase):
    checkpoint_format = "torchtitan"
    default_output_dir = "evaluations/simple-evals"

    @classmethod
    def run(cls, shared_args: SharedArgs, *, own_args: argparse.Namespace, **kwargs):
        from .simple_evals import run_simple_evals, prepare_simple_evals

        shared_args.output_dir = shared_args.output_dir or cls.default_output_dir

        final_args = own_args

        final_args.config = shared_args.model_config
        final_args.checkpoint = shared_args.model_checkpoint
        final_args.model = "custom-model"

        # Construct path relative to the checkpoint directory
        checkpoint_base_dir = shared_args.model_checkpoint
        if os.path.isfile(shared_args.model_checkpoint):
            checkpoint_base_dir = os.path.dirname(shared_args.model_checkpoint)
        final_args.output_folder = os.path.join(
            checkpoint_base_dir, shared_args.output_dir
        )
        models, grading_sampler, equality_checker = prepare_simple_evals(final_args)
        return run_simple_evals(final_args, models, grading_sampler, equality_checker)

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        from .simple_evals import get_simple_evals_parser

        # get_simple_evals_parser returns a new parser. We need to add its arguments
        # to the existing subparser, excluding those that are already global.
        simple_evals_parser_obj = get_simple_evals_parser()

        # Arguments to exclude as they are handled by the global parser (shared_args)
        excluded_dests = ["output_folder", "config", "checkpoint", "model"]

        for action in simple_evals_parser_obj._actions:
            # Avoid adding help action again if it's the default one.
            if action.dest == "help" and isinstance(action, argparse._HelpAction):
                continue
            # Skip arguments that are handled globally
            if action.dest in excluded_dests:
                continue

            parser._add_action(action)
        parser.description = "Run Simple-Evals benchmarks. Uses the custom model by default via global --model-config and --model-checkpoint arguments."
        return parser


EVALUATION_METHODS = {
    "olmes": OlmesEval,
    "eleuther": EleutherEval,
    "prompt": GenerateFromPrompt,
    "simple-evals": SimpleEval,
}
