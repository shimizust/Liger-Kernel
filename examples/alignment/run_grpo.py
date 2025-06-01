from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import psutil
import time
from transformers import TrainerCallback
import logging
import csv
import os
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryProfilingCallback(TrainerCallback):
    def __init__(self, output_dir, profile_steps=[], use_torch_profiler=True):
        self.output_dir = output_dir
        self.process = psutil.Process()
        self.profile_steps = profile_steps
        self.profiler = None
        self.use_torch_profiler = use_torch_profiler
        os.makedirs(output_dir, exist_ok=True)
        
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            # Start recording memory history
            torch.cuda.memory._record_memory_history(max_entries=100000)
            
    def on_step_begin(self, args, state, control, **kwargs):
        # Only profile on the main process (rank 0)
        if state.global_step in self.profile_steps and args.local_rank <= 0:
            logger.info(f"Starting profiler for step {state.global_step} on rank {args.local_rank}")
            if self.use_torch_profiler:
                # Start profiling at the beginning of each step
                self.profiler = profile(
                    activities=[ProfilerActivity.CUDA],  # Only track GPU memory
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=0,
                        active=1,
                        repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        os.path.join(self.output_dir, f"local_rank_{args.local_rank}_steps_{self.profile_steps[0]}_to_{self.profile_steps[-1]}")
                    ),
                    profile_memory=True,  # Keep memory profiling
                    with_stack=False,     # Disable stack traces
                    with_flops=False,     # Disable FLOP counting
                    with_modules=False,   # Disable module tracking
                )
                self.profiler.start()
                logger.info(f"Profiler started for step {state.global_step}")
        
    def on_step_end(self, args, state, control, **kwargs):
        # Add debug logging at the start of the method
        logger.info(f"on_step_end called for step {state.global_step}, local_rank {args.local_rank}")
        logger.info(f"Profile steps: {self.profile_steps}, Current step in profile steps: {state.global_step in self.profile_steps}")
        
        # Only process profiling data on the main process (rank 0)
        if state.global_step in self.profile_steps and args.local_rank <= 0:
            if self.use_torch_profiler:
                logger.info(f"Attempting to stop profiler for step {state.global_step} on rank {args.local_rank}")
                if self.profiler is None:
                    logger.warning(f"Profiler is None for step {state.global_step} on rank {args.local_rank}")
                    return
                    
                # Stop profiling and save results
                self.profiler.stop()
                logger.info(f"Profiler stopped for step {state.global_step}")
            
            # Log basic memory stats for all GPUs
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_allocated(i)
                    gpu_peak = torch.cuda.max_memory_allocated(i)
                    logger.info(f"Step {state.global_step} - GPU {i} Memory: {gpu_memory/1024/1024:.2f}MB, Peak: {gpu_peak/1024/1024:.2f}MB")
            
            # Save memory snapshot
            try:
                logger.info("Attempting to save memory snapshot...")
                snapshot_path = os.path.join(self.output_dir, f"memory_snapshot_step_{state.global_step}.pickle")
                torch.cuda.memory._dump_snapshot(snapshot_path)
                logger.info(f"Memory snapshot saved to {snapshot_path}")
            except Exception as e:
                logger.error(f"Failed to save memory snapshot: {e}")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Print key events from the trace if using torch profiler
            if self.use_torch_profiler:
                logger.info(f"\nStep {state.global_step} Memory Profile:")
                for evt in self.profiler.key_averages():
                    if evt.key in ['forward', 'backward', 'optimizer_step']:
                        logger.info(
                            f"{evt.key}: "
                            f"GPU Memory: {evt.cuda_memory_usage/1024/1024:.2f}MB, "
                            f"Duration: {evt.cpu_time_total/1000:.2f}ms"
                        )
    
    def on_train_end(self, args, state, control, **kwargs):
        # Stop recording memory history
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("Memory history recording stopped")

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

def run_training(use_liger=True, max_steps=5):
    output_dir = f"/home/azureuser/sshimizu/outputs/grpo-qwen2.5-0.5b-instruct-{'with' if use_liger else 'without'}-liger"
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_steps=1,  # Log every step for better memory tracking
        use_liger_loss=use_liger,
        per_device_train_batch_size=4,
        num_generations=4,
        max_steps=max_steps,  # Limit the number of training steps
    )
    
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Add memory profiling callback with output directory and steps to profile
    trainer.add_callback(MemoryProfilingCallback(output_dir, profile_steps=[2], use_torch_profiler=False))
    
    # Run training
    trainer.train()
    
    logger.info(f"Detailed memory profiles saved to: {output_dir}")

# Run training with and without Liger for just a few steps
logger.info("Starting training without Liger kernel (5 steps)...")
run_training(use_liger=False, max_steps=5)

logger.info("\nStarting training with Liger kernel (5 steps)...")
run_training(use_liger=True, max_steps=5)