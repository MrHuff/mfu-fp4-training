#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from transformers import (
    LlamaTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import os
from datasets import load_dataset, Dataset
import torch
import time  # Import the time module
from functools import wraps  # Import wraps for decorators

import multiprocessing
from tqdm import tqdm
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState
from benchmark_models.utils import BF16LossScaler
from torch.profiler import profile, ProfilerActivity

from cut_cross_entropy.transformers import cce_patch


# Add this new class to your script
class ProfilerStepCallback(TrainerCallback):
    """A callback that calls profiler.step() at the end of each training step."""

    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()


# --- Timing Decorator ---
def timeit(func):
    """A simple decorator to measure and print the execution time of a function."""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        print(f"⏱️  Starting '{func.__name__}'...")
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        total_time = end_time - start_time
        print(f"✅ Finished '{func.__name__}' in {total_time:.4f} seconds.")
        return result

    return timeit_wrapper


# Mocking the low_bits_training imports for standalone execution if they are not installed
try:
    from low_bits_training.quantization.fusedMXFPMatmul import swap_llama_mlp_to_mxlinear
    from low_bits_training.quantization.stable_spam import StableSPAM
except ImportError:
    print(
        "Warning: 'low_bits_training' library not found. Quantization functions will be skipped."
    )

    def swap_linear_with_mx_linear_fused(*args, **kwargs):
        print("Mock: swap_linear_with_mx_linear_fused called.")
        pass

    class StableSPAM:
        def __init__(self, *args, **kwargs):
            print("Mock: StableSPAM initialized.")
            pass


# Your existing model configurations
model9M = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 128,
    "intermediate_size": 352,
    "initializer_range": 0.02,
    "max_sequence_length": 128,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "pad_token_id": -1,  # This will be corrected at runtime
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32000,
    "parameter_count": 9_000_000,
    "batch_size": 4096,
    "learning_rate": 1e-3,
}

model60M = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 512,
    "intermediate_size": 1376,
    "initializer_range": 0.02,
    "max_sequence_length": 512,
    "model_type": "llama",
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "pad_token_id": -1,  # This will be corrected at runtime
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32000,
    "parameter_count": 60_000_000,
    "batch_size": 256,
    "learning_rate": 1e-3,
}

model350M = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 2736,
    "initializer_range": 0.02,
    "max_sequence_length": 1024,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": -1,  # This will be corrected at runtime
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32000,
    "parameter_count": 350_000_000,
    "batch_size": 64,
    "learning_rate": 1e-4,
}
model1B = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 5461,
    "initializer_range": 0.02,
    "max_sequence_length": 1024,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "pad_token_id": -1,  # This will be corrected at runtime
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32000,
    "parameter_count": 1_000_000_000,
    "batch_size": 16,
    "learning_rate": 1e-4,
}

model7B = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "initializer_range": 0.02,
    "max_sequence_length": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "pad_token_id": -1,  # This will be corrected at runtime
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32000,
    "parameter_count": 7_000_000_000,
    "batch_size": 4,
    "learning_rate": 1e-4,
}


class StopOnNaNCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        if loss is not None and (loss != loss):  # NaN check
            print(f"❌ NaN loss detected at step {state.global_step}. Stopping training.")
            control.should_training_stop = True


# --- NEW: Custom Callback for Verbose Logging ---
class VerboseLoggingCallback(TrainerCallback):
    """A custom callback to print detailed training statistics at each log step."""

    def __init__(self, logging_steps: int):
        super().__init__()
        self.logging_steps = logging_steps
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        print("🚀 Starting training...")
        self.last_log_time = time.monotonic()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if state.is_world_process_zero and logs:
            current_time = time.monotonic()
            time_since_last_log = current_time - self.last_log_time

            # Extract metrics from the logs dictionary
            loss = logs.get("loss")
            learning_rate = logs.get("learning_rate")
            epoch = logs.get("epoch")

            if loss is not None:
                # Calculate average time per step since the last log
                avg_step_time = time_since_last_log / self.logging_steps

                # Format and print the log string
                log_str = (
                    f"Step: {state.global_step:5d} | "
                    f"Loss: {loss:.4f} | "
                    f"LR: {learning_rate:.2e} | "
                    f"Epoch: {epoch:.2f} | "
                    f"Time/Step: {avg_step_time:.3f}s"
                )
                print(log_str)

            self.last_log_time = current_time


# --- Custom Trainer Class (Unchanged) ---
class CustomOptimizerTrainer(Trainer):
    def __init__(
        self,
        *args,
        optimizer_eval_string: str = None,
        lr=1e-3,
        use_loss_scaler=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_eval_string = optimizer_eval_string
        self.lr = lr
        self.loss_scaler = BF16LossScaler()
        self.use_loss_scaler = use_loss_scaler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        lr = self.lr
        if self.optimizer_eval_string:
            try:
                self.optimizer = eval(self.optimizer_eval_string, globals(), locals())
            except Exception as e:
                raise ValueError(
                    f"Error evaluating optimizer_eval_string: '{self.optimizer_eval_string}'. "
                    f"Make sure it's valid Python code and necessary variables/imports are accessible. Error: {e}"
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )

        from transformers import get_linear_schedule_with_warmup

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.optimizer, self.lr_scheduler

    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if not torch.isfinite(loss):
            print(
                f"❌ Non-finite loss detected at step {self.state.global_step}. Stopping training."
            )
            self.control.should_training_stop = True
            return loss.detach()
        if self.use_loss_scaler:
            scaled_loss = self.loss_scaler.scale_loss(loss)
            scaled_loss.backward()
            self.loss_scaler.unscale_grads(model)
        else:
            loss.backward()
        return loss.detach()

    def optimizer_step(self, model, optimizer, scheduler=None):
        if self.use_loss_scaler:
            self.loss_scaler.step(optimizer)
        else:
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()


def truncate_token_lists(n_tokens, nested_lists):
    flat = []
    for sublist in tqdm(nested_lists, desc="Truncating tokens"):
        needed = n_tokens - len(flat)
        if needed <= 0:
            break
        flat.extend(sublist[:needed])
    return flat


class LlamaTrainerWrapper:
    CONFIG_MAP = {
        "llama_9M": model9M,
        "llama_60M": model60M,
        "llama_350M": model350M,
        "llama_1B": model1B,
        "llama_7B": model7B,
    }

    def __init__(
        self,
        model_type: str,
        pretrained_tokenizer_name: str = "NousResearch/Llama-2-7b-hf",
        dataset_name: str = "wikipedia",
        dataset_config: str = "20220301.en",
        eval_interval: int = 1000,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        optimizer_eval_string: str = None,
        num_dataloader_workers: int = 20,
        loss_scaling: bool = False,
    ):
        config_dict = self.CONFIG_MAP.get(model_type)

        self.model_type = model_type
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_tokenizer_name)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.loss_scaling = loss_scaling
        param_count = config_dict.get("parameter_count")
        if param_count is None:
            raise ValueError(
                "Model config must include 'parameter_count' to compute training tokens."
            )
        self.max_tokens = param_count * 100
        self.sequence_length = config_dict["max_sequence_length"]
        self.batch_size = config_dict["batch_size"]
        self.eval_interval = eval_interval
        self.learning_rate = config_dict.get("learning_rate")
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.optimizer_eval_string = optimizer_eval_string
        self.num_dataloader_workers = num_dataloader_workers
        self.output_dir = f"./output_{model_type}"

        self.train_dataset = None
        self.val_dataset = None
        self.model = None
        self.trainer = None
        # Use a sensible number of processes for data prep, not tied to dataloader_num_workers
        self.num_proc = multiprocessing.cpu_count()

    @timeit
    def load_and_tokenize_data(self):
        # This function is unchanged but will benefit from timing decorator
        cached_train_path = (
            f"tokenized_train_seq{self.sequence_length}_max{self.max_tokens}.arrow"
        )
        cached_test_path = (
            f"tokenized_test_seq{self.sequence_length}_max{self.max_tokens}.arrow"
        )

        if os.path.exists(cached_train_path) and os.path.exists(cached_test_path):
            t_start_cache = time.monotonic()
            print(f"Loading cached train dataset from {cached_train_path}...")
            train_ds = Dataset.load_from_disk(cached_train_path)
            print(f"Loading cached test dataset from {cached_test_path}...")
            test_ds = Dataset.load_from_disk(cached_test_path)
            t_end_cache = time.monotonic()
            print(
                f"    -> Loading from cache took {t_end_cache - t_start_cache:.4f} seconds."
            )

            epochs_needed = float(self.max_tokens * (0.9)) / float(
                len(train_ds) * self.sequence_length
            )
            self.num_epochs = max(1, round(epochs_needed))
            print(f"Total epochs for training: {self.num_epochs}")
            return train_ds, test_ds

        t_start_processing = time.monotonic()

        t_start = time.monotonic()
        try:
            raw_ds = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed to load with config: {e}. Retrying without config...")
            raw_ds = load_dataset(
                self.dataset_name, split="train", trust_remote_code=True
            )
        print(
            f"    -> Loading raw dataset took {time.monotonic() - t_start:.4f} seconds."
        )

        article_count = min(self.max_tokens // 750, 6458670)
        raw_ds = raw_ds.shuffle(seed=42)
        raw_ds = raw_ds.select(range(article_count))

        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], truncation=False)

        t_start = time.monotonic()
        tokenized_ds = raw_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            num_proc=self.num_proc,
            load_from_cache_file=False,
        )
        print(
            f"    -> Tokenizing raw dataset took {time.monotonic() - t_start:.4f} seconds."
        )

        t_start = time.monotonic()
        flat_token_ds = tokenized_ds.map(
            lambda ex: {"tokens": ex["input_ids"]},
            batched=True,
            remove_columns=tokenized_ds.column_names,
            num_proc=self.num_proc,
            load_from_cache_file=False,
        )
        print(
            f"    -> Mapping to token IDs took {time.monotonic() - t_start:.4f} seconds."
        )

        t_start = time.monotonic()
        token_stream = truncate_token_lists(self.max_tokens, flat_token_ds["tokens"])
        print(f"Total flattened tokens: {len(token_stream)}")
        print(
            f"    -> Flattening and truncating token stream took {time.monotonic() - t_start:.4f} seconds."
        )

        epochs_needed = float(self.max_tokens) / float(len(token_stream))
        self.num_epochs = max(1, round(epochs_needed))
        print(f"Total epochs: {self.num_epochs}")

        t_start = time.monotonic()
        token_stream_np = np.array(token_stream, dtype=np.int32)
        eos_id = self.tokenizer.eos_token_id
        seq_len = self.sequence_length
        total_chunks = token_stream_np.shape[0] // seq_len
        if total_chunks == 0:
            input_ids = np.full((1, seq_len), eos_id, dtype=np.int32)
        else:
            trimmed = token_stream_np[: total_chunks * seq_len]
            input_ids = trimmed.reshape((total_chunks, seq_len))
        labels = input_ids.copy()
        print(
            f"    -> Chunking tokens into sequences took {time.monotonic() - t_start:.4f} seconds."
        )

        n_chunks = input_ids.shape[0]
        train_end_idx = int(n_chunks * (0.9))

        t_start = time.monotonic()
        train_chunked_ds = Dataset.from_dict(
            {
                "input_ids": input_ids[:train_end_idx].tolist(),
                "labels": labels[:train_end_idx].tolist(),
            }
        )
        test_chunked_ds = Dataset.from_dict(
            {
                "input_ids": input_ids[train_end_idx:].tolist(),
                "labels": labels[train_end_idx:].tolist(),
            }
        )
        print(
            f"    -> Creating HF datasets from chunks took {time.monotonic() - t_start:.4f} seconds."
        )

        t_start = time.monotonic()
        train_chunked_ds.save_to_disk(cached_train_path)
        test_chunked_ds.save_to_disk(cached_test_path)
        print(
            f"    -> Saving datasets to disk took {time.monotonic() - t_start:.4f} seconds."
        )

        print(
            f"    -> Total data processing time: {time.monotonic() - t_start_processing:.4f} seconds."
        )

        return train_chunked_ds, test_chunked_ds

    @timeit
    def load_model(self):
        config_dict = self.CONFIG_MAP.get(self.model_type)
        if config_dict is None:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Choose from {list(self.CONFIG_MAP.keys())}"
            )

        config = LlamaConfig(**config_dict)
        self.model = LlamaForCausalLM(config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @timeit
    def setup_trainer(self):
        """
        Sets up the Hugging Face Trainer with more verbose logging.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Define logging steps here to pass to the callback
        logging_steps = 10

        args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            # --- MODIFIED FOR VERBOSITY ---
            logging_steps=logging_steps,  # Log more frequently
            disable_tqdm=False,  # Show the progress bar
            log_level="info",  # Show info-level logs from transformers
            # --- END MODIFICATIONS ---
            report_to="none",
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            dataloader_num_workers=self.num_dataloader_workers,
            load_best_model_at_end=False,
            bf16=True,
            save_strategy="no",
            eval_strategy="no",
            remove_unused_columns=False,
        )

        # Instantiate and add the new verbose callback
        verbose_callback = VerboseLoggingCallback(logging_steps=logging_steps)

        self.trainer = CustomOptimizerTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=self.train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            optimizer_eval_string=self.optimizer_eval_string,
            lr=self.learning_rate,
            use_loss_scaler=self.loss_scaling,
            # Add the new callback to the list
            callbacks=[StopOnNaNCallback(), verbose_callback],
        )

    # @timeit
    # def train(self):
    #     """
    #     Starts the training process. The output will now be verbose.
    #     """
    #     self.trainer.train()
    @timeit
    def debug_train(self):
        """
        Starts the training process, wrapped with the PyTorch profiler.
        """
        log_dir = f"./logs/{self.model_type}_profile"

        # This schedule will skip the first 5 steps, warm up for 1,
        # and then actively record the next 3 steps.
        prof_schedule = torch.profiler.schedule(wait=5, warmup=1, active=3, repeat=1)

        # The handler tells the profiler to save the output for TensorBoard.
        trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)

        print(f"🚀 Starting training with profiler. Trace will be saved to '{log_dir}'")
        print(
            "Profiler will wait 5 steps, warmup for 1, and then record steps 7, 8, and 9."
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,  # Optional: records tensor shapes
            with_stack=True,  # Optional: records source code locations
        ) as prof:
            # Add our custom callback to the trainer to signal the profiler
            self.trainer.add_callback(ProfilerStepCallback(profiler=prof))

            # Run the training as usual
            self.trainer.train()

        print(
            f"\n✅ Profiling finished. To view the trace, run:\n    tensorboard --logdir {log_dir}"
        )

    @timeit
    def train(self):
        """
        Starts the training process, wrapped with the PyTorch profiler.
        """

        self.trainer.train()

    def run(self, config=None):
        """
        Executes the full Llama training pipeline, now with torch.compile.
        """
        print("--- LlamaTrainerWrapper Run Started ---")

        self.train_dataset, self.val_dataset = self.load_and_tokenize_data()
        print(f"Training dataset size: {len(self.train_dataset)}")

        self.load_model()

        # Apply quantization if a config is provided
        if config is not None:
            print("Applying quantization via swap_linear_with_mx_linear_fused...")
            swap_llama_mlp_to_mxlinear(self.model, config=config, include_lm_head=False)
            print("Quantization applied.")
        else:
            print("No quantization config provided. Training without quantization.")
        print(self.model)
        # --- COMPILE THE MODEL HERE ---
        print("🚀 Compiling the Llama model with torch.compile...")
        try:
            # "reduce-overhead" is a safe and effective mode for complex models.
            self.model = cce_patch(self.model)
            self.model = torch.compile(self.model)
            print("✅ Model compiled successfully.")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}. Running without compilation.")
        # --- END OF CHANGE ---

        self.setup_trainer()
        self.train()

        # Extracting losses (code is unchanged)
        print("\n--- Extracting Training and Validation Losses ---")
        training_losses = []
        validation_losses = []

        for log_entry in self.trainer.state.log_history:
            if "loss" in log_entry:
                training_losses.append(
                    {
                        "step": log_entry.get("step"),
                        "epoch": log_entry.get("epoch"),
                        "loss": log_entry["loss"],
                    }
                )
            if "eval_loss" in log_entry:
                validation_losses.append(
                    {
                        "step": log_entry.get("step"),
                        "epoch": log_entry.get("epoch"),
                        "eval_loss": log_entry["eval_loss"],
                    }
                )

        print(f"Found {len(training_losses)} training loss entries.")
        print("--- Loss Extraction Complete ---")

        return training_losses, validation_losses
