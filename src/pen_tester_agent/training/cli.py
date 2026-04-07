"""CLI entry point for MLX GRPO fine-tuning on Apple Silicon."""

import argparse

from .config import TrainingConfig
from .trainer import run_training


def main():
    """Run the MLX GRPO fine-tuning pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a small LLM for penetration testing agent tasks using GRPO on Apple Silicon.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="MLX model name or path (default: 3B, or 0.5B with --low-memory)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (default: 16, or 8 with --low-memory)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TrainingConfig.max_steps,
        help="Maximum training steps (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: 1024, or 512 with --low-memory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingConfig.learning_rate,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TrainingConfig.temperature,
        help="Sampling temperature for GRPO generations (default: %(default)s)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of generations per prompt for GRPO (default: 4, or 2 with --low-memory)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=TrainingConfig.dataset_size,
        help="Number of training examples (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=TrainingConfig.output_dir,
        help="Output directory for model checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=TrainingConfig.clip_eps,
        help="PPO clipping epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=TrainingConfig.kl_coeff,
        help="KL divergence penalty coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=TrainingConfig.grad_accumulation_steps,
        help="Gradient accumulation steps (default: %(default)s)",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Fuse LoRA and export for Ollama after training",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode for machines with 8GB RAM or less. "
             "Uses 0.5B model instead of 3B, reduces generations (4→2), "
             "sequence length (1024→512), and LoRA rank (16→8).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print generated completions and reward breakdowns each step",
    )

    args = parser.parse_args()

    # Build config: start from low-memory preset or defaults.
    if args.low_memory:
        # Collect explicit overrides from CLI args.
        overrides = {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "dataset_size": args.dataset_size,
            "output_dir": args.output_dir,
            "clip_eps": args.clip_eps,
            "kl_coeff": args.kl_coeff,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "verbose": args.verbose,
        }
        # Only override preset values if explicitly provided.
        if args.model is not None:
            overrides["model_name"] = args.model
        if args.lora_rank is not None:
            overrides["lora_rank"] = args.lora_rank
        if args.max_seq_length is not None:
            overrides["max_seq_length"] = args.max_seq_length
        if args.num_generations is not None:
            overrides["num_generations"] = args.num_generations
        config = TrainingConfig.low_memory_preset(**overrides)
    else:
        config = TrainingConfig(
            model_name=args.model or TrainingConfig.model_name,
            lora_rank=args.lora_rank or TrainingConfig.lora_rank,
            max_steps=args.max_steps,
            max_seq_length=args.max_seq_length or TrainingConfig.max_seq_length,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            num_generations=args.num_generations or TrainingConfig.num_generations,
            dataset_size=args.dataset_size,
            output_dir=args.output_dir,
            clip_eps=args.clip_eps,
            kl_coeff=args.kl_coeff,
            grad_accumulation_steps=args.grad_accumulation_steps,
            verbose=args.verbose,
        )

    run_training(config, export_gguf=args.export_gguf)


if __name__ == "__main__":
    main()
