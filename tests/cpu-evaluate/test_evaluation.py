#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
import argparse


from low_bits_training.evaluation import (
    EvaluationBase,
    get_shared_parser,
    EVALUATION_METHODS,
)


@pytest.mark.parametrize("method_name", EVALUATION_METHODS.keys())
def test_evaluation_method_parsing_robust(method_name):
    """
    Tests parser identifies the method, handles required shared args,
    retrieves a valid EvaluationProtocol class, and checks for the
    presence of method-specific args without relying on defaults.
    """
    parser = get_shared_parser()
    evaluation_class = EVALUATION_METHODS[method_name]

    # --- Basic Setup: Provide only strictly required args ---
    test_config = f"dummy/{method_name}_config.yaml"
    test_checkpoint = f"dummy/{method_name}_checkpoint"
    # Output dir might have a default, but let's provide it for consistency
    test_output = f"test/output/{method_name}"

    # Base required arguments for the main parser
    args_list = [
        "--model-config",
        test_config,
        "--model-checkpoint",
        test_checkpoint,
        "--output-dir",
        test_output,
        method_name,
    ]

    # TODO: Add any *required* args specific to a subparser
    if method_name == "prompt":
        args_list.append("--prompt")
        args_list.append("Hello, world!")

    # --- Parse Arguments ---
    args = parser.parse_args(args_list)

    # 1. Correct method selected?
    assert args.method == method_name

    # 2. Required shared arguments parsed?
    assert hasattr(args, "model_config") and args.model_config == test_config
    assert hasattr(args, "model_checkpoint") and args.model_checkpoint == test_checkpoint
    assert hasattr(
        args, "output_dir"
    )  # Check presence, value might be default if not provided

    # 3. Correct Evaluation Class retrieved and is it valid?
    assert issubclass(evaluation_class, EvaluationBase)

    # 4. Protocol methods/attributes exist?
    assert hasattr(evaluation_class, "run") and callable(evaluation_class.run)
    assert hasattr(evaluation_class, "get_parser") and callable(
        evaluation_class.get_parser
    )
    assert hasattr(evaluation_class, "checkpoint_format")
    assert evaluation_class.checkpoint_format in ["torchtitan", "transformers"]

    # 5. Check PRESENCE of optional args defined by the method's parser
    #    This confirms the subparser ran and registered its args, without
    #    being brittle about specific default values.
    temp_subparser = argparse.ArgumentParser(add_help=False)  # Use temporary parser
    evaluation_class.get_parser(temp_subparser)
    method_specific_arg_names = {action.dest for action in temp_subparser._actions}

    # Check that the final parsed args object has attributes for these names
    for arg_name in method_specific_arg_names:
        # Ignore shared args already checked and the method itself
        if arg_name not in ["model_config", "model_checkpoint", "output_dir", "method"]:
            assert hasattr(args, arg_name), (
                f"Method '{method_name}': Parsed args object is missing "
                f"expected attribute '{arg_name}' defined by its get_parser."
            )


def test_olmes_wandb_flag_behavior():
    """Tests only the --wandb flag default and activation for olmes."""
    parser = get_shared_parser()
    base_args = ["--model-config", "cfg", "--model-checkpoint", "ckpt", "olmes"]

    # Test default (wandb flag not provided)
    args_default = parser.parse_args(base_args)
    assert hasattr(args_default, "wandb"), "Olmes args missing 'wandb'"
    assert args_default.wandb is False, "Olmes 'wandb' default should be False"

    # Test activation (wandb flag provided)
    args_activated = parser.parse_args(base_args + ["--wandb"])
    assert hasattr(args_activated, "wandb"), "Olmes args missing 'wandb'"
    assert args_activated.wandb is True, "Olmes 'wandb' should be True when flag is set"
