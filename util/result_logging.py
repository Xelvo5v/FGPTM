import json
from pathlib import Path

import numpy as np

import util.utils as utils


METRIC_FILES = ("epoch_metrics.jsonl", "best_metrics.json", "log.txt")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def build_epoch_record(args, epoch, stats, n_parameters):
    record = {
        "epoch": int(epoch),
        "epoch_display": int(epoch) + 1,
        "n_parameters": int(n_parameters),
        "ptm_type": args.ptm_type,
        "repeat": args.repeat,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_embed_dim": args.seq_embed_dim,
        "width": args.width,
        "num_heads": args.num_heads,
        "moe_num_experts": args.moe_num_experts,
        "seed": args.seed,
    }
    record.update({key: json_safe(value) for key, value in stats.items()})
    return record


def reset_metric_files(output_dir):
    if not output_dir or not utils.is_main_process():
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename in METRIC_FILES:
        metric_path = output_path / filename
        if metric_path.exists():
            metric_path.unlink()


def write_epoch_metrics(args, epoch, stats, n_parameters):
    if not args.output_dir or not utils.is_main_process():
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = build_epoch_record(args, epoch, stats, n_parameters)
    with (output_dir / "epoch_metrics.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_best_metrics(args, best_record):
    if not args.output_dir or not utils.is_main_process() or not best_record:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "best_metrics.json").open("w") as f:
        f.write(json.dumps(best_record, indent=2, sort_keys=True) + "\n")


def is_better_metrics(candidate, current):
    if current is None:
        return True
    candidate_key = (
        candidate.get("auprc", float("-inf")),
        candidate.get("auc", float("-inf")),
        -candidate.get("loss", float("inf")),
    )
    current_key = (
        current.get("auprc", float("-inf")),
        current.get("auc", float("-inf")),
        -current.get("loss", float("inf")),
    )
    return candidate_key > current_key
