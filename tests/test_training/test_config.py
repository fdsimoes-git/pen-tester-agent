"""Tests for training configuration."""

from pen_tester_agent.training.config import TrainingConfig


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.model_name == "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
        assert config.max_seq_length == 1024
        assert config.lora_rank == 16
        assert config.max_steps == 250
        assert config.output_dir == "outputs"
        assert config.clip_eps == 0.2
        assert config.kl_coeff == 0.04

    def test_custom_values(self):
        config = TrainingConfig(
            model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
            lora_rank=8,
            max_steps=100,
        )
        assert config.model_name == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert config.lora_rank == 8
        assert config.max_steps == 100

    def test_grpo_params(self):
        config = TrainingConfig()
        assert config.num_generations == 4
        assert config.grad_accumulation_steps == 4
        assert config.sync_interval == 10
        assert config.temperature == 0.8

    def test_low_memory_default_false(self):
        config = TrainingConfig()
        assert config.low_memory is False

    def test_low_memory_preset(self):
        config = TrainingConfig.low_memory_preset()
        assert config.low_memory is True
        assert config.model_name == "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
        assert config.num_generations == 2
        assert config.max_seq_length == 512
        assert config.lora_rank == 8
        # Other defaults should remain unchanged.
        assert config.max_steps == 250
        assert config.learning_rate == 1e-6

    def test_low_memory_preset_with_overrides(self):
        config = TrainingConfig.low_memory_preset(
            max_steps=100,
            num_generations=3,
        )
        assert config.low_memory is True
        assert config.num_generations == 3  # overridden
        assert config.max_seq_length == 512  # preset default
        assert config.max_steps == 100  # overridden
