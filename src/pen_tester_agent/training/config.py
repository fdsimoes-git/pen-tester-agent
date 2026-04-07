"""Training configuration for MLX GRPO fine-tuning on Apple Silicon."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters and model configuration for MLX GRPO training."""

    # Model
    model_name: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    max_seq_length: int = 1024
    lora_rank: int = 16

    # GRPO training
    num_generations: int = 4
    max_steps: int = 250
    learning_rate: float = 1e-6
    temperature: float = 0.8
    clip_eps: float = 0.2
    kl_coeff: float = 0.04
    grad_accumulation_steps: int = 4
    sync_interval: int = 10
    num_epochs: int = 1

    # Output
    output_dir: str = "outputs"
    save_steps: int = 50
    logging_steps: int = 1

    # Dataset
    dataset_size: int = 500

    # Memory
    low_memory: bool = False

    # Verbosity
    verbose: bool = False

    @classmethod
    def low_memory_preset(cls, **overrides) -> "TrainingConfig":
        """Return a config tuned for machines with 8GB RAM or less.

        Uses a smaller 0.5B model, reduces num_generations, max_seq_length,
        and lora_rank to fit comfortably in 8GB unified memory.
        """
        defaults = {
            "low_memory": True,
            "model_name": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
            "num_generations": 2,
            "max_seq_length": 512,
            "lora_rank": 8,
        }
        defaults.update(overrides)
        return cls(**defaults)
