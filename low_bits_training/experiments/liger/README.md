# Liger Kernel Experiment

This experiment replaces standard PyTorch implementations with LinkedIn's [Liger](https://github.com/linkedin/Liger-Kernel) re-implementations as Triton kernels for improved performance during training.

## Currently Supported
- Cross-entropy loss

## Usage
```
CONFIG_FILE=train_configs/llama3_8b_profiling.toml ./run_train.sh # baseline
CONFIG_FILE=train_configs/llama3_8b_profiling.toml ./run_train.sh --job.experimental_modules=liger # replace cross-entropy
```

## Result
The baseline memory usage peaks at 43.3 GB, with the cross-entropy Liger kernel it peaks at 40.9 GB. Other performance metrics are unchanged.
