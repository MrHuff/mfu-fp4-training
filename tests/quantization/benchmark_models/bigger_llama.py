from transformers import (
    LlamaTokenizer, LlamaConfig, LlamaForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
import os
import torch
from torch import optim
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from typing import Optional, Tuple

import json
import time
from functools import wraps
import pickle
import math
import multiprocessing
from tqdm import tqdm
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState
from benchmark_models.utils import BF16LossScaler
from torch.profiler import profile, ProfilerActivity
from transformers import TrainerCallback
from cut_cross_entropy.transformers import cce_patch
import torch.distributed as dist
# --- MODIFIED: Added get_last_checkpoint for robust resumption ---
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction, EvalLoopOutput, has_length, denumpify_detensorize
from transformers.trainer_pt_utils import EvalLoopContainer, IterableDatasetShard,find_batch_size
from transformers.utils import is_torch_xla_available, logging
from typing import Optional,List
from transformers.integrations.deepspeed import deepspeed_init
from transformers.models.llama.modeling_llama import LlamaMLP

logger = logging.get_logger(__name__)

class ScaledSwiglu:
    # def __call__(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     x = torch.chunk(x, 2, dim=-1)
    #     s = x[1].detach().abs().max(dim=-1, keepdim=True)[0]
    #     tmp = x[1] / s
    #     return torch.nn.functional.silu(x[0]) * tmp
    def __call__(self, gate:torch.Tensor, up_proj:torch.Tensor):
        s = up_proj.detach().abs().max(dim=-1, keepdim=True)[0]
        tmp =up_proj / s
        return torch.nn.functional.silu(gate) * tmp


def patch_llama_mlp_SSWIGLU(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            # Replace its forward with one using ScaledSwiglu
            old_gate, old_up, old_down = module.gate_proj, module.up_proj, module.down_proj
            act = ScaledSwiglu()

            def new_forward(x, gate=old_gate, up=old_up, down=old_down, act=act):
                # mimic concatenation before your ScaledSwiglu
                h = act(gate(x),up(x))
                return down(h)  # ignore s here unless you want to propagate it
            module.forward = new_forward

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
        # In DDP, only print from the main process to avoid cluttered logs
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"⏱️  Starting '{func.__name__}'...")
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        total_time = end_time - start_time
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"✅ Finished '{func.__name__}' in {total_time:.4f} seconds.")
        return result
    return timeit_wrapper

# Mocking the low_bits_training imports for standalone execution if they are not installed
try:
    from low_bits_training.quantization.fusedMXFPMatmul import swap_llama_mlp_to_mxlinear
    from low_bits_training.quantization.stable_spam import StableSPAM
except ImportError:
    print("Warning: 'low_bits_training' library not found. Quantization functions will be skipped.")
    def swap_linear_with_mx_linear_fused(*args, **kwargs):
        print("Mock: swap_linear_with_mx_linear_fused called.")
        pass
    class StableSPAM:
        def __init__(self, *args, **kwargs):
            print("Mock: StableSPAM initialized.")
            pass

# Model configurations (unchanged)
model9M = { "architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 128, "intermediate_size": 352, "initializer_range": 0.02, "max_sequence_length": 128, "model_type": "llama", "num_attention_heads": 4, "num_hidden_layers": 4, "pad_token_id": -1, "rms_norm_eps": 1e-06, "transformers_version": "4.28.1", "use_cache": True, "vocab_size": 32000, "parameter_count":  9_000_000, "batch_size": 4096, "learning_rate": 1e-3,'gradient_accumulation_steps': 1 }
model60M = { "architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 512, "intermediate_size": 1376, "initializer_range": 0.02, "max_sequence_length": 512, "model_type": "llama", "num_attention_heads": 8, "num_hidden_layers": 8, "pad_token_id": -1, "rms_norm_eps": 1e-06, "transformers_version": "4.28.1", "use_cache": True, "vocab_size": 32000, "parameter_count":  60_000_000, "batch_size": 128, "learning_rate": 1e-4,'gradient_accumulation_steps': 1 }
model350M = { "architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 1024, "intermediate_size": 2720, "initializer_range": 0.02, "max_sequence_length": 1024, "model_type": "llama", "num_attention_heads": 16, "num_hidden_layers": 24, "pad_token_id": -1, "rms_norm_eps": 1e-06, "transformers_version": "4.28.1", "use_cache": True, "vocab_size": 32000, "parameter_count":  350_000_000, "batch_size": 16, "learning_rate": 1e-4, 'gradient_accumulation_steps': 8}
model1B = { "architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 5472, "initializer_range": 0.02, "max_sequence_length": 1024, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 24, "pad_token_id": -1, "rms_norm_eps": 1e-06, "transformers_version": "4.28.1", "use_cache": True, "vocab_size": 32000, "parameter_count":  1_000_000_000, "batch_size": 4, "learning_rate": 1e-4,'gradient_accumulation_steps': 512}
model7B = { "architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "pad_token_id": -1, "rms_norm_eps": 1e-06, "transformers_version": "4.28.1", "use_cache": True, "vocab_size": 32000, "parameter_count":  7_000_000_000, "batch_size": 4, "learning_rate": 1e-4,'gradient_accumulation_steps': 32 }

class StopOnNaNCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        if loss is not None and (loss != loss):
            print(f"❌ NaN loss detected at step {state.global_step}. Stopping training.")
            control.should_training_stop = True

# --- MODIFIED: VerboseLoggingCallback to handle both training and validation logs ---
class VerboseLoggingCallback(TrainerCallback):
    def __init__(self, logging_steps: int):
        super().__init__()
        self.logging_steps = logging_steps
        self.last_log_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print("🚀 Starting training...")
            self.last_log_time = time.monotonic()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        current_time = time.monotonic()
        
        # Check if this is an evaluation log
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            eval_log_str = (
                f"Step: {state.global_step:5d} | "
                f"Validation Loss: {eval_loss:.4f} | "
                f"Epoch: {logs.get('eval_epoch', 0):.2f}"
            )
            print(f"\n{'='*25} VALIDATION {'='*25}\n{eval_log_str}\n{'='*62}\n")
            return # Don't process as a training log

        # Otherwise, process as a training log
        train_loss = logs.get("loss")
        if train_loss is not None:
            time_since_last = current_time - (self.last_log_time if self.last_log_time else current_time)
            avg_step_time = (time_since_last / self.logging_steps) if self.last_log_time else 0.0

            log_str = (
                f"Step: {state.global_step:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {logs.get('learning_rate'):.2e} | "
                f"Epoch: {logs.get('epoch'):.2f} | "
                f"Time/Step: {avg_step_time:.3f}s"
            )
            print(log_str)
            self.last_log_time = current_time

class CustomOptimizerTrainer(Trainer):
    def __init__(self, *args, optimizer_eval_string: str = None, lr = 1e-3, use_loss_scaler=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_eval_string = optimizer_eval_string
        self.lr = lr
        self.use_loss_scaler = use_loss_scaler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        lr = self.lr 
        if self.optimizer_eval_string:
            try:
                self.optimizer = eval(self.optimizer_eval_string, globals(), locals())
            except Exception as e:
                raise ValueError(f"Error evaluating optimizer_eval_string: '{self.optimizer_eval_string}'. Error: {e}")
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        
        from transformers import get_linear_schedule_with_warmup
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return self.optimizer, self.lr_scheduler

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_batch_sizes = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            tensor_observed_batch_size = torch.tensor(observed_batch_size, device=self.args.device)
            global_observed_batch_size =  self.gather_function((tensor_observed_batch_size))
            all_batch_sizes.add(global_observed_batch_size)

            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)

            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()
        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_batch_sizes = all_batch_sizes.get_arrays()
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_batch_sizes
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

   
    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if not torch.isfinite(loss):
             if self.args.local_rank == 0:
                print(f"❌ Non-finite loss detected at step {self.state.global_step}. Stopping training.")
             self.control.should_training_stop = True
             return loss.detach()
        self.accelerator.backward(loss)
        return loss.detach()

def truncate_token_lists(n_tokens, nested_lists):
    flat = []
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    iterable = tqdm(nested_lists, desc="Truncating tokens", disable=not is_main_process)
    for sublist in iterable:
        needed = n_tokens - len(flat)
        if needed <= 0:
            break
        flat.extend(sublist[:needed])
    return flat


def compute_metrics(eval_pred):
    weights = eval_pred.inputs
    logits = eval_pred.predictions
    """Computes cross-entropy loss from the model's logits."""
    return {
        "eval_loss": (weights * logits).sum().item()/weights.sum().item()
    }
class LlamaTrainerWrapperBigger:
    CONFIG_MAP = {
        'llama_9M': model9M, 'llama_60M': model60M, 'llama_350M': model350M,
        'llama_1B': model1B, 'llama_7B': model7B, 
        'llama_9M_SSWIG': model9M,
        'llama_60M_SSWIG': model60M,
        'llama_350M_SSWIG': model350M,
        'llama_1B_SSWIG': model1B,
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
        loss_scaling: bool = False,
        num_dataloader_workers: int = 0,
        checkpoint_folder: str = 'checkpoint_folder'
    ):
        config_dict = self.CONFIG_MAP.get(model_type)
        self.model_type = model_type
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_tokenizer_name)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.loss_scaling = loss_scaling
        param_count = config_dict.get("parameter_count")
        self.max_tokens = param_count * 100 if model_type in ['llama_9M','llama_60M','llama_60M_SSWIG','llama_9M_SSWIG'] else param_count*42
        self.sequence_length = config_dict['max_sequence_length']
        self.batch_size = config_dict['batch_size']
        self.eval_interval = eval_interval
        self.learning_rate = config_dict.get('learning_rate')
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.optimizer_eval_string = optimizer_eval_string
        cwd = os.getcwd()
        self.output_dir = os.path.join(cwd, f'output_{checkpoint_folder}_{model_type}')
        self.train_dataset = None
        self.val_dataset = None
        self.model = None
        self.trainer = None
        self.gradient_accumulation_steps = config_dict.get('gradient_accumulation_steps')
        self.num_proc = max(1, multiprocessing.cpu_count() - 2)

    @timeit
    def load_and_tokenize_data(self):
        is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
        cached_train_path = f"tokenized_train_seq{self.sequence_length}.arrow"
        cached_test_path = f"tokenized_test_seq{self.sequence_length}.arrow"

        if os.path.exists(cached_train_path) and os.path.exists(cached_test_path):
            if is_main_process:
                print(f"✅ Found pre-processed data cache.")
        else:
            # This block handles one-time data processing if the cache is not found.
            # The validation set created here will be the full size.
            if is_main_process:
                print("⚠️ No cached data found. Processing data from scratch on main process...")
                try:
                    raw_ds = load_dataset(self.dataset_name, self.dataset_config, split='train', trust_remote_code=True)
                except Exception as e:
                    print(f"Failed to load with config: {e}. Retrying without config...")
                    raw_ds = load_dataset(self.dataset_name, split='train', trust_remote_code=True)

                def tokenize_fn(examples):
                    return self.tokenizer(examples["text"], truncation=False)

                tokenized_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=self.num_proc, load_from_cache_file=False)
                flat_token_ds = tokenized_ds.map(lambda ex: {"tokens": ex["input_ids"]}, batched=True, remove_columns=tokenized_ds.column_names, num_proc=self.num_proc, load_from_cache_file=False)
                token_stream = truncate_token_lists(self.max_tokens, flat_token_ds["tokens"])
                token_stream_np = np.array(token_stream, dtype=np.int32)
                eos_id = self.tokenizer.eos_token_id
                seq_len = self.sequence_length
                total_chunks = token_stream_np.shape[0] // seq_len
                input_ids = token_stream_np[:total_chunks * seq_len].reshape((total_chunks, seq_len)) if total_chunks > 0 else np.full((1, seq_len), eos_id, dtype=np.int32)
                labels = input_ids.copy()
                n_chunks = input_ids.shape[0]
                train_end_idx = int(n_chunks * 0.9)
                train_chunked_ds = Dataset.from_dict({"input_ids": input_ids[:train_end_idx].tolist(), "labels": labels[:train_end_idx].tolist()})
                test_chunked_ds = Dataset.from_dict({"input_ids": input_ids[train_end_idx:].tolist(), "labels": labels[train_end_idx:].tolist()})

                train_chunked_ds.save_to_disk(cached_train_path)
                test_chunked_ds.save_to_disk(cached_test_path)
                print("✅ Data processing and caching complete.")
            
            # All other processes wait for the main process to finish caching.
            if dist.is_initialized():
                dist.barrier()
        
        # All processes load the datasets from the cache.
        if is_main_process:
            print("💾 Loading datasets from disk...")
        self.train_dataset = Dataset.load_from_disk(cached_train_path)
        self.val_dataset = Dataset.load_from_disk(cached_test_path)
        
        # --- MODIFIED: Shrink the validation dataset on-the-fly after loading ---
        original_val_size = len(self.val_dataset)
        shrink_factor = 10
        new_val_size = max(1, original_val_size // shrink_factor)
        self.val_dataset = self.val_dataset.select(range(new_val_size))
        # --- END OF MODIFICATION ---
        
        epochs_needed = float(self.max_tokens * 0.9) / float(len(self.train_dataset) * self.sequence_length)
        self.num_epochs = max(1, epochs_needed)
        if is_main_process:
            print(f"Total epochs for training: {self.num_epochs:.2f}")
            print(f"Validation dataset shrunk from {original_val_size} to {len(self.val_dataset)} samples.")
        
        return self.train_dataset, self.val_dataset

    @timeit
    def load_model(self):
        config_dict = self.CONFIG_MAP.get(self.model_type)
        config = LlamaConfig(**config_dict)
        self.model = LlamaForCausalLM(config)

        if 'SSWIG' in self.model_type:
            patch_llama_mlp_SSWIGLU(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'}) 
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @timeit
    def setup_trainer(self):
        os.makedirs(self.output_dir, exist_ok=True)
        logging_steps = 10
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        # --- 1. Define your training parameters ---
        # Assuming you have these defined or passed in as variables
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        gradient_accumulation_steps = self.gradient_accumulation_steps
        # Let's assume train_dataset is your loaded training dataset
        # train_dataset = ... 
        num_training_samples = len(self.train_dataset) 
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # --- 3. Calculate total training steps ---
        # The formula is the same, but now `world_size` is correct for your DDP setup
        steps_per_epoch = math.ceil(num_training_samples / (batch_size * world_size * gradient_accumulation_steps))
        total_training_steps = steps_per_epoch * num_epochs

        # --- 4. Calculate the 10% interval ---
        logging_and_eval_steps = max(1, int(0.1 * total_training_steps))

        print(f"Total GPUs (World Size): {world_size}")
        print(f"Total training steps: {total_training_steps}")
        print(f"Evaluating and saving every {logging_and_eval_steps} steps.")

        # --- 4. Initialize TrainingArguments with the calculated value ---
        args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            # Use the calculated value here
            logging_steps=logging_and_eval_steps,
            save_steps=logging_and_eval_steps,
            eval_steps=logging_and_eval_steps,
            # Set evaluation strategy to 'steps'
            save_strategy="steps",
            eval_strategy="steps",
            # Other parameters...
            disable_tqdm=False,
            log_level="info",
            report_to="none",
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            bf16=True,
            ddp_find_unused_parameters=False,
            local_rank=local_rank,
            remove_unused_columns=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_total_limit=2,
            per_device_eval_batch_size=batch_size
        )
        
        verbose_callback = VerboseLoggingCallback(logging_steps=logging_steps)

        self.trainer = CustomOptimizerTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset, # --- ADDED: Pass validation dataset
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            optimizer_eval_string=self.optimizer_eval_string,
            lr=self.learning_rate,
            use_loss_scaler=self.loss_scaling,
            callbacks=[StopOnNaNCallback(), verbose_callback],
            compute_metrics=compute_metrics
        )

    # --- MODIFIED: train() method for robust checkpoint handling ---
    @timeit
    def train(self):
        """Checks for a checkpoint and resumes if found, otherwise starts fresh."""
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        last_checkpoint = get_last_checkpoint(self.trainer.args.output_dir)

        if last_checkpoint is not None:
            if is_main_process:
                print(f"🤝 Checkpoint found at {last_checkpoint}. Resuming training.")
            resume_arg = last_checkpoint
        else:
            if is_main_process:
                print("🏁 No checkpoint found. Starting training from scratch.")
            resume_arg = False
            
        self.trainer.train(resume_from_checkpoint=resume_arg)

    def run(self, config=None):
        if "LOCAL_RANK" in os.environ and not dist.is_initialized():
            cached_train_path = f"tokenized_train_seq{self.sequence_length}.arrow"
            import datetime
            timeout = datetime.timedelta(minutes=10) if os.path.exists(cached_train_path) else datetime.timedelta(minutes=1200) 
            dist.init_process_group(backend='nccl', timeout=timeout)
        
        is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
        if is_main_process:
            print("--- LlamaTrainerWrapper Run Started (DDP Mode) ---")

        self.train_dataset, self.val_dataset = self.load_and_tokenize_data()

        self.load_model()

        if config is not None:
            if is_main_process:
                print("Applying quantization via swap_linear_with_mx_linear_fused...")
            swap_llama_mlp_to_mxlinear(self.model, config=config, include_lm_head=False)
            if is_main_process:
                print("Quantization applied.")
        
        if is_main_process:
            print(self.model)

        if is_main_process:
            print("🚀 Compiling the Llama model with torch.compile...")
        try:
            self.model = cce_patch(self.model)
            if is_main_process:
                print("✅ CCE patch applied to compiled model.")
            self.model = torch.compile(self.model)
            if is_main_process:
                print("✅ Model compiled successfully.")
        except Exception as e:
            if is_main_process:
                print(f"⚠️ torch.compile or patch failed: {e}. Running without compilation.")

        self.setup_trainer()
        self.train()
        # Extracting losses (code is unchanged)
        print("\n--- Extracting Training and Validation Losses ---")
        training_losses = []
        validation_losses = []
        if is_main_process:
            for log_entry in self.trainer.state.log_history:
                if 'loss' in log_entry:
                    training_losses.append({
                        'step': log_entry.get('step'),
                        'epoch': log_entry.get('epoch'),
                        'loss': log_entry['loss']
                    })
                if 'eval_loss' in log_entry:
                    validation_losses.append({
                        'step': log_entry.get('step'),
                        'epoch': log_entry.get('epoch'),
                        'eval_loss': log_entry['eval_loss']
                    })
            
            print(f"Found {len(training_losses)} training loss entries.")
            print(f"Found {len(validation_losses)} val loss entries.")
            print("--- Loss Extraction Complete ---")

            return training_losses, validation_losses