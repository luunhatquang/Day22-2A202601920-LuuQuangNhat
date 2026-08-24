from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


def score_response(prompt: str, response: str) -> float:
    """Return a deterministic CPU-only relevance and fluency score.

    This heuristic deliberately receives no preference label.  It scores every
    response with the same prompt-overlap, lexical-diversity, and length rules,
    making it suitable for a reproducible smoke-test evaluation when no model is
    available.  It is not a replacement for model log-probabilities in production.
    """
    prompt_tokens = set(re.findall(r"\w+", prompt.lower()))
    resp_tokens = re.findall(r"\w+", response.lower())

    if not resp_tokens:
        return -10.0

    # Relevance to the prompt intent.
    overlap_ratio = len(set(resp_tokens) & prompt_tokens) / max(1, len(prompt_tokens))
    lexical_diversity = len(set(resp_tokens)) / len(resp_tokens)
    useful_length = min(len(resp_tokens), 60) / 60
    excess_length_penalty = 0.01 * max(0, len(resp_tokens) - 80)

    return round(
        float(0.6 * overlap_ratio + 0.25 * lexical_diversity + 0.15 * useful_length - excess_length_penalty),
        4,
    )


def split_examples_for_evaluation(cfg: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Load data and create the deterministic prompt-level train/validation split."""
    examples = load_jsonl(cfg["paths"]["train_data"])
    evaluation = cfg.get("evaluation", {})
    return split_by_prompt(
        examples,
        validation_ratio=float(evaluation.get("validation_ratio", 0.2)),
        seed=int(cfg.get("seed", 42)),
    )


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
        ),
    ] = Path("configs/local.yaml"),
) -> None:
    cfg = load_config(config)
    train_section = cfg.get("training", {})
    paths_section = cfg.get("paths", {})

    train_cfg = TrainingConfig(
        method=train_section.get("method", "dpo"),
        beta=float(train_section.get("beta", 0.1)),
        lambda_orpo=float(train_section.get("lambda_orpo", 0.1)),
        max_length=int(train_section.get("max_length", 512)),
        batch_size=int(train_section.get("batch_size", 2)),
        output_dir=paths_section.get("output_dir", "outputs"),
    )

    train_examples, validation_examples = split_examples_for_evaluation(cfg)
    trainer = PreferenceTrainer(config=train_cfg, examples=train_examples)
    metrics = trainer.train()

    print("[green]Training completed successfully![/green]")
    print(f"[blue]Method:[/blue] {metrics['method']}")
    print(
        f"[blue]Initial Loss:[/blue] {metrics['initial_loss']} -> [blue]Final Loss:[/blue] {metrics['final_loss']}"
    )
    print(f"[blue]Held-out validation examples:[/blue] {len(validation_examples)}")
    print(f"[green]Artifacts saved to {train_cfg.output_dir}[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
        ),
    ] = Path("configs/local.yaml"),
) -> None:
    cfg = load_config(config)
    train_examples, examples = split_examples_for_evaluation(cfg)

    chosen_scores = [score_response(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [score_response(ex.prompt, ex.rejected) for ex in examples]

    acc = pairwise_accuracy(examples, chosen_scores, rejected_scores)
    mean_chosen = round(sum(chosen_scores) / len(chosen_scores), 4) if chosen_scores else 0.0
    mean_rejected = (
        round(sum(rejected_scores) / len(rejected_scores), 4) if rejected_scores else 0.0
    )
    margin = round(mean_chosen - mean_rejected, 4)

    metrics: dict[str, Any] = {
        "pairwise_accuracy": acc,
        "mean_chosen_score": mean_chosen,
        "mean_rejected_score": mean_rejected,
        "preference_margin": margin,
        "train_examples": len(train_examples),
        "total_examples": len(examples),
    }

    # Evaluate safety regression prompts if configured
    regression_path = cfg.get("evaluation", {}).get("regression_prompts")
    if regression_path and Path(regression_path).exists():
        content = Path(regression_path).read_text(encoding="utf-8")
        prompts = [
            line.strip() for line in content.splitlines() if re.match(r"^\d+\.", line.strip())
        ]
        metrics["safety_regression_prompts_checked"] = len(prompts)
        metrics["safety_regression_status"] = "manual_review_required"

    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")
    print(f"[blue]Pairwise Accuracy:[/blue] {acc * 100:.1f}%")
    print(f"[blue]Preference Margin:[/blue] {margin}")


if __name__ == "__main__":
    app()
