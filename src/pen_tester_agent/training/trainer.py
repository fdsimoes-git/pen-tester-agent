"""GRPO training orchestration using MLX on Apple Silicon.

Implements Group Relative Policy Optimization with a custom training loop
using mlx and mlx-lm — no PyTorch, no CUDA dependencies.

Memory-efficient: uses a single base model with LoRA weight swapping
instead of keeping three full model copies in RAM.
"""

import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .config import TrainingConfig
from .rewards import (
    command_quality_reward,
    explanation_reward,
    format_reward,
    tool_selection_reward,
)
from .datasets import build_dataset

REWARD_FUNCS = [
    format_reward,
    tool_selection_reward,
    command_quality_reward,
    explanation_reward,
]

# LoRA target layer keys (standard transformer attention + MLP).
LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def _get_lora_weights(model) -> list:
    """Extract only LoRA adapter parameters (lora_a, lora_b) from the model.

    Returns a deep copy of the LoRA weight pairs, independent of future
    model updates.  Only copies the tiny adapter matrices (~few MB), NOT
    the base model weights.
    """
    pairs = [
        (k, mx.array(v))
        for k, v in tree_flatten(model.trainable_parameters())
        if "lora_a" in k or "lora_b" in k
    ]
    mx.eval([v for _, v in pairs])
    return pairs


def _clear_cache():
    """Release unused GPU memory back to the system (no-op on CPU)."""
    if mx.default_device() == mx.cpu:
        return
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        # Best-effort cache clear: failures are non-fatal and can be safely ignored.
        pass


def _swap_lora_weights(model, weights: list):
    """Load a saved set of LoRA adapter weights into the model."""
    model.load_weights(weights, strict=False)
    # Only evaluate the swapped LoRA weights, not the entire model.
    mx.eval([v for _, v in weights])


def load_model(config: TrainingConfig):
    """Load a model + tokenizer from MLX-community and apply LoRA.

    Uses a single base model with separate LoRA weight snapshots for
    the rollout policy and frozen reference, instead of deep-copying
    the entire model three times.

    Returns:
        Tuple of (model, old_weights, ref_weights, tokenizer) where:
        - model: trainable policy with LoRA adapters
        - old_weights: LoRA weight snapshot for rollout policy
        - ref_weights: frozen LoRA weight snapshot for KL penalty
        - tokenizer: the tokenizer
    """
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(config.model_name)

    # Apply LoRA using mlx-lm's built-in utility.
    num_layers = len(model.model.layers)
    lora_config = {
        "rank": config.lora_rank,
        "scale": config.lora_rank * 2.0,
        "dropout": 0.0,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, num_layers, lora_config)

    # Freeze base model weights so only LoRA adapters are trainable.
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])

    # Snapshot the initial LoRA weights for reference and rollout.
    # Only copies lora_a/lora_b matrices (~few MB), not base model weights.
    ref_weights = _get_lora_weights(model)
    old_weights = _get_lora_weights(model)
    _clear_cache()

    # Ensure pad token exists.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, old_weights, ref_weights, tokenizer


def generate_completions(model, tokenizer, prompt_text: str,
                         config: TrainingConfig) -> list[str]:
    """Generate multiple completions for a prompt using the rollout model.

    Args:
        model: The model (caller should swap in rollout weights first).
        tokenizer: The tokenizer.
        prompt_text: Formatted prompt string.
        config: Training config with temperature and num_generations.

    Returns:
        List of completion strings.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    completions = []
    max_tokens = config.max_seq_length // 2
    sampler = make_sampler(temp=config.temperature)

    for _ in range(config.num_generations):
        output = generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        # Some generate() APIs return the full prompt+completion text.
        # Ensure we only keep the newly generated completion portion.
        if isinstance(output, str) and output.startswith(prompt_text):
            output = output[len(prompt_text):]
        completions.append(output)

    return completions


def compute_log_probs(model, tokenizer, prompt_text: str,
                      completion: str) -> mx.array:
    """Compute log-probability of a completion given a prompt.

    Uses vectorized indexing to extract only the needed log-probs,
    avoiding Python loops over the large logits tensor.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt_text: The prompt string.
        completion: The completion string.

    Returns:
        Scalar mx.array with the summed log-probability.
    """
    prompt_tokens = tokenizer.encode(prompt_text)
    completion_tokens = tokenizer.encode(completion)
    full_tokens = prompt_tokens + completion_tokens

    input_ids = mx.array(full_tokens)[None, :]  # (1, seq_len)
    logits = model(input_ids)  # (1, seq_len, vocab_size)

    # Build index arrays for the completion token positions.
    prompt_len = len(prompt_tokens)
    positions = []
    token_ids = []
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1  # logit at position t predicts token t+1
        if 0 <= pos < logits.shape[1]:
            positions.append(pos)
            token_ids.append(token_id)

    if not positions:
        return mx.array(0.0)

    # Extract only the logits at positions we need, then log-softmax.
    pos_logits = logits[0, mx.array(positions), :]  # (n_tokens, vocab_size)
    log_probs = pos_logits - mx.logsumexp(pos_logits, axis=-1, keepdims=True)
    total_log_prob = mx.sum(log_probs[mx.arange(len(token_ids)), mx.array(token_ids)])

    return total_log_prob


def _format_prompt(tokenizer, messages: list[dict]) -> str:
    """Format messages using the tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    # Fallback for tokenizers without chat template.
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def compute_advantages(rewards_per_completion: list[float]) -> list[float]:
    """Compute normalized advantages from raw rewards.

    GRPO normalizes within the group: A_i = (R_i - mean) / (std + eps)
    """
    n = len(rewards_per_completion)
    if n == 0:
        return []
    mean_r = sum(rewards_per_completion) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards_per_completion) / n
    std_r = var_r ** 0.5
    eps = 1e-8
    return [(r - mean_r) / (std_r + eps) for r in rewards_per_completion]


def train_step(model, old_weights, ref_weights, tokenizer,
               prompt_messages, config: TrainingConfig,
               current_weights=None):
    """Execute one GRPO training step for a single prompt.

    Uses LoRA weight swapping to avoid keeping three full models in memory.
    Pre-computes old/ref log-probs outside the gradient tape to reduce
    peak memory during backprop.

    Returns:
        Tuple of (loss_value, gradients, metrics_dict).
    """
    prompt_text = _format_prompt(tokenizer, prompt_messages)

    # Save current training weights before swapping.
    if current_weights is None:
        current_weights = _get_lora_weights(model)

    # 1. Generate completions using the rollout model (old weights).
    _swap_lora_weights(model, old_weights)
    completions = generate_completions(model, tokenizer, prompt_text, config)
    _clear_cache()

    # 2. Score completions with all reward functions.
    formatted = [[{"content": c}] for c in completions]
    total_rewards = [0.0] * len(completions)
    reward_breakdown = {fn.__name__: [] for fn in REWARD_FUNCS}

    for reward_fn in REWARD_FUNCS:
        kwargs = {}
        if reward_fn.__name__ == "tool_selection_reward":
            kwargs["prompts"] = [prompt_messages] * len(completions)
        scores = reward_fn(formatted, **kwargs)
        reward_breakdown[reward_fn.__name__] = scores
        for i, s in enumerate(scores):
            total_rewards[i] += s

    # 3. Compute advantages.
    advantages = compute_advantages(total_rewards)

    # 4. Compute old log-probs (under rollout model) — outside gradient tape.
    #    old_weights are already loaded from step 1.
    old_log_probs = []
    for comp in completions:
        lp = compute_log_probs(model, tokenizer, prompt_text, comp)
        old_log_probs.append(lp)
    mx.eval(old_log_probs)
    _clear_cache()

    # 5. Compute ref log-probs (under reference model) — outside gradient tape.
    _swap_lora_weights(model, ref_weights)
    ref_log_probs = []
    for comp in completions:
        lp = compute_log_probs(model, tokenizer, prompt_text, comp)
        ref_log_probs.append(lp)
    mx.eval(ref_log_probs)
    _clear_cache()

    # 6. Restore current training weights for gradient computation.
    _swap_lora_weights(model, current_weights)

    # 7. Define loss function — only new_log_probs computed inside gradient tape.
    def loss_fn():
        total_loss = mx.array(0.0)
        count = 0

        for i, comp in enumerate(completions):
            if abs(advantages[i]) < 1e-10:
                continue

            new_lp = compute_log_probs(model, tokenizer, prompt_text, comp)
            old_lp = old_log_probs[i]
            ref_lp = ref_log_probs[i]

            # Policy ratio.
            ratio = mx.exp(new_lp - old_lp)
            adv = mx.array(advantages[i])

            # Clipped PPO objective.
            clipped = mx.clip(ratio, 1.0 - config.clip_eps,
                              1.0 + config.clip_eps)
            ppo_obj = mx.minimum(ratio * adv, clipped * adv)

            # KL penalty (approximation: r - log(r) - 1).
            log_ratio_kl = ref_lp - new_lp
            ratio_kl = mx.exp(log_ratio_kl)
            kl = ratio_kl - log_ratio_kl - 1.0

            total_loss = total_loss - ppo_obj + config.kl_coeff * kl
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    # 8. Compute gradients.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_val, grads = loss_and_grad_fn()

    # Extract the user task from the prompt for logging.
    user_task = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            user_task = msg["content"]
            break

    metrics = {
        "loss": loss_val.item(),
        "mean_reward": sum(total_rewards) / len(total_rewards),
        "max_reward": max(total_rewards),
        "min_reward": min(total_rewards),
        "completions": completions,
        "rewards": total_rewards,
        "reward_breakdown": reward_breakdown,
        "advantages": advantages,
        "task": user_task,
    }

    return loss_val, grads, metrics


def sync_models(model) -> list:
    """Snapshot current LoRA weights for use as rollout model."""
    return _get_lora_weights(model)


def _configure_memory(config: TrainingConfig):
    """Configure MLX device and memory for the training environment.

    In low-memory mode, switches to CPU to avoid Metal GPU memory pressure
    from other processes (IDE, browser, OS). On Apple Silicon the CPU still
    uses the unified memory pool but bypasses Metal's command-buffer
    allocator, which is the typical OOM failure point on 8 GB machines.
    """
    if not config.low_memory:
        return
    mx.set_default_device(mx.cpu)
    print("Low-memory mode: using CPU backend (Metal GPU bypassed)")


def run_training(config: TrainingConfig, export_gguf: bool = False):
    """End-to-end MLX GRPO training pipeline.

    Args:
        config: Training configuration.
        export_gguf: Whether to fuse and export after training.
    """
    _configure_memory(config)
    print(f"Loading model: {config.model_name}")
    model, old_weights, ref_weights, tokenizer = load_model(config)

    print(f"Building dataset ({config.dataset_size} examples)...")
    dataset = build_dataset(size=config.dataset_size)

    optimizer = optim.AdamW(learning_rate=config.learning_rate)

    print(f"Starting GRPO training for {config.max_steps} steps...")
    print(f"  Generations per prompt: {config.num_generations}")
    print(f"  Clip epsilon: {config.clip_eps}")
    print(f"  KL coefficient: {config.kl_coeff}")
    print(f"  Gradient accumulation: {config.grad_accumulation_steps}")
    if config.low_memory:
        print(f"  Low-memory mode: ON (seq={config.max_seq_length}, "
              f"rank={config.lora_rank}, gens={config.num_generations})")
    print()

    step = 0
    accumulated_grads = None

    for _epoch in range(config.num_epochs):
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for idx in indices:
            if step >= config.max_steps:
                break

            item = dataset[idx]
            prompt_messages = item["prompt"]

            t0 = time.time()
            _loss_val, grads, metrics = train_step(
                model, old_weights, ref_weights, tokenizer,
                prompt_messages, config,
            )
            dt = time.time() - t0

            # Accumulate gradients.
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g, accumulated_grads, grads,
                )

            # Apply gradients after accumulation window.
            if (step + 1) % config.grad_accumulation_steps == 0:
                scaled_grads = tree_map(
                    lambda g: g / config.grad_accumulation_steps,
                    accumulated_grads,
                )
                optimizer.update(model, scaled_grads)
                mx.eval(model.parameters(), optimizer.state)
                accumulated_grads = None

            # Sync rollout model periodically.
            if (step + 1) % config.sync_interval == 0:
                old_weights = sync_models(model)

            step += 1

            # Logging.
            if step % config.logging_steps == 0:
                print(
                    f"[Step {step}/{config.max_steps}] "
                    f"loss={metrics['loss']:.4f}  "
                    f"reward={metrics['mean_reward']:.2f} "
                    f"(min={metrics['min_reward']:.1f} "
                    f"max={metrics['max_reward']:.1f})  "
                    f"time={dt:.1f}s"
                )

                if config.verbose:
                    print(f"  Task: {metrics['task']}")
                    for j, comp in enumerate(metrics["completions"]):
                        print(f"\n  --- Generation {j+1} "
                              f"(reward={metrics['rewards'][j]:.1f}, "
                              f"adv={metrics['advantages'][j]:.2f}) ---")
                        # Per-reward breakdown.
                        parts = []
                        for name, scores in metrics["reward_breakdown"].items():
                            short = name.replace("_reward", "")
                            parts.append(f"{short}={scores[j]:+.1f}")
                        print(f"  Rewards: {', '.join(parts)}")
                        # Truncate long completions for readability.
                        text = comp.strip()
                        if len(text) > 500:
                            text = text[:500] + "..."
                        for line in text.split("\n"):
                            print(f"  | {line}")
                    print()

            # Save checkpoint.
            if step % config.save_steps == 0:
                ckpt_dir = Path(config.output_dir) / f"step-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_file = str(ckpt_dir / "adapters.safetensors")
                model.save_weights(ckpt_file)
                print(f"  Checkpoint saved to {ckpt_file}")

        if step >= config.max_steps:
            break

    # Final save.
    print(f"\nSaving final adapter to {config.output_dir}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(str(Path(config.output_dir) / "adapters.safetensors"))

    if export_gguf:
        export_model(config)

    print("Training complete.")


def export_model(config: TrainingConfig):
    """Fuse LoRA adapter and export for Ollama.

    Uses mlx_lm fuse to merge the adapter into the base model,
    then provides instructions for GGUF conversion.
    """
    import subprocess

    fused_path = f"{config.output_dir}/fused"
    print("Fusing LoRA adapter into base model...")

    subprocess.run([
        "mlx_lm.fuse",
        "--model", config.model_name,
        "--adapter-path", config.output_dir,
        "--save-path", fused_path,
        "--de-quantize",
    ], check=True)

    print(f"Fused model saved to {fused_path}")
    print()
    print("To convert to GGUF for Ollama:")
    print(f"  1. python llama.cpp/convert_hf_to_gguf.py {fused_path}"
          " --outfile pen-tester-agent.gguf")
    print("  2. ollama create pen-tester-agent-finetuned -f Modelfile")
    print()
    print("Modelfile contents:")
    print("  FROM ./pen-tester-agent.gguf")
