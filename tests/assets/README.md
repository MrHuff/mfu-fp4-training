# Test Assets

This directory contains assets used for testing purposes.

## Checkpoint

Place a distributed checkpoint in the `checkpoint` directory. This checkpoint should be a small LLaMA model checkpoint in the distributed checkpoint (DCP) format. The checkpoint will be used for testing the checkpoint conversion functionality.

### Expected Files

A typical DCP checkpoint will include:
- `metadata.json` - Contains metadata about the checkpoint
- `__0_0.distcp` - Shard files containing the actual model weights
- Other shard files depending on the distribution configuration

### How to Generate a Test Checkpoint

If you need to generate a test checkpoint, you can do it with a GPU by using the create_seed_checkpoint flag.

```bash
NGPU=1 CONFIG=train_configs/debug_model.toml WANDB_MODE=disabled WANDB_NAME=REPLACE ./run_train.sh  --checkpoint.enable_checkpoint --checkpoint.create_seed_checkpoint --training.data_parallel_replicate_degree 1 --training.data_parallel_shard_degree 1 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 1 --experimental.context_parallel_degree 1 --job.dump_folder tests/assets/
```

This will create a new run with a checkpoint from step 0.

This checkpoint can be converted to a standard PyTorch checkpoint using the following command:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch tests/assets/checkpoint/step-0/ tests/assets/checkpoint/collated_checkpoint.pt
```

This can be converted back to a DCP checkpoint using the following command:

```bash
python -m torch.distributed.checkpoint.format_utils torch_to_dcp tests/assets/checkpoint/collated_checkpoint.pt tests/assets/checkpoint/step-0/
```

The checkpoint in either format contains the metadata about the shard shapes and ranks.

## Handling changes in checkpoint formats

When the checkpoints change format you should provide backward compatibility behind the `--checkpoint.compatibility` flag.

Rename the old test checkpoint to: `test-checkpoint-unit_test-lbt-vXX`. Add a test case for your new compatibility flag in `tests/gpu/test_checkpoints_gpu.py::test_load_checkpoint_compat`.
