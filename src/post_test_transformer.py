"""Run the fixed AfroXLMR-mini CPU follow-up without opening the old test split."""

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data import LABELS, ROOT, load_split, manifest
from src.evaluate import evaluate_predictions
from src.post_test_weighting import read_json, sha256


PLAN_PATH = ROOT / "configs/post_test_transformer.json"
REPORT_PATH = ROOT / "reports/post_test_transformer.json"


def validate_plan(path=PLAN_PATH):
    plan = read_json(path)
    fixed = (
        "post_test_afroxlmr_mini_v1", "Davlan/afro-xlmr-mini",
        "bc04038b969667884bd83cdd37bed9559c2c3d9c", 42, 96, 3, 8, 2,
        0.00002, 0.01, "balanced from full train label frequencies", 8,
    )
    actual = tuple(plan[k] for k in (
        "experiment_id", "base_model", "base_model_revision", "seed",
        "max_token_length", "epochs", "batch_size", "gradient_accumulation_steps",
        "learning_rate", "weight_decay", "class_weight", "cpu_threads"))
    if actual != fixed or plan["label_order"] != list(LABELS):
        raise ValueError("The predeclared transformer protocol changed.")
    if plan["selection_metric"] != (
            "No hyperparameter selection; evaluate final third epoch on development once"):
        raise ValueError("The development evaluation protocol changed.")
    for filename, key in [
        ("reports/final_test.json", "original_final_test_report_sha256"),
        ("reports/post_test_weighting.json", "prior_weighting_report_sha256"),
        ("reports/post_test_hybrid.json", "prior_hybrid_report_sha256"),
    ]:
        if sha256(ROOT / filename) != plan[key]:
            raise ValueError(f"Prior report changed: {filename}")
    source = manifest()
    if source["revision"] != plan["dataset_revision"]:
        raise ValueError("Pinned dataset revision changed.")
    for split in ("train", "dev"):
        entry = next(item for item in source["files"]
                     if item["local_path"] == f"data/raw/amh/{split}.tsv")
        if entry["sha256"] != plan[f"{split}_sha256"]:
            raise ValueError(f"Pinned {split} checksum changed.")
    return plan


def run(plan_path=PLAN_PATH, report_path=REPORT_PATH, artifact_root=ROOT / "artifacts"):
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError("This transformer experiment already has a report.")
    plan = validate_plan(plan_path)
    # Import optional, large dependencies only after checking the declared protocol.
    import numpy as np
    import torch
    import transformers
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(plan["cpu_threads"])
    torch.set_num_interop_threads(1)
    train = load_split("train")
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    counts = Counter(row["label"] for row in train)
    weights = torch.tensor([len(train) / (len(LABELS) * counts[label])
                            for label in LABELS], dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        plan["base_model"], revision=plan["base_model_revision"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        plan["base_model"], revision=plan["base_model_revision"],
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=label_to_id)
    model.to("cpu")
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"],
                                  weight_decay=plan["weight_decay"])

    def collate(batch):
        encoded = tokenizer([row["tweet"] for row in batch], padding=True,
                            truncation=True, max_length=plan["max_token_length"],
                            return_tensors="pt")
        targets = torch.tensor([label_to_id[row["label"]] for row in batch],
                               dtype=torch.long)
        return encoded, targets

    loader = DataLoader(train, batch_size=plan["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(plan["seed"]),
                        collate_fn=collate, num_workers=0)
    training_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(plan["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, row_count = 0.0, 0
        for step, (inputs, labels) in enumerate(loader, start=1):
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            loss_sum += float(loss.detach()) * len(labels)
            row_count += len(labels)
            (loss / plan["gradient_accumulation_steps"]).backward()
            if step % plan["gradient_accumulation_steps"] == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch={epoch + 1}/{plan['epochs']} step={step}/{len(loader)}",
                      flush=True)
        epoch_losses.append(loss_sum / row_count)
        print(f"epoch={epoch + 1} training_loss={epoch_losses[-1]:.6f}", flush=True)

    training_seconds = time.perf_counter() - training_start
    # Development is opened only after the fixed final epoch, for one evaluation.
    dev = load_split("dev")
    eval_loader = DataLoader(dev, batch_size=plan["batch_size"], shuffle=False,
                             collate_fn=collate, num_workers=0)
    model.eval()
    predicted = []
    with torch.inference_mode():
        for inputs, _ in eval_loader:
            predicted.extend(LABELS[int(i)] for i in model(**inputs).logits.argmax(dim=1))
    metrics = evaluate_predictions([row["label"] for row in dev], predicted)
    artifact_dir = Path(artifact_root) / (
        "post_test_transformer_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_" + uuid4().hex[:6])
    artifact_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(artifact_dir, safe_serialization=True)
    tokenizer.save_pretrained(artifact_dir)
    model_file = artifact_dir / "model.safetensors"
    prior = read_json(ROOT / "reports/post_test_weighting.json")
    report = {
        "experiment_id": plan["experiment_id"], "status": plan["status"],
        "plan_sha256": sha256(plan_path), "code_sha256": sha256(Path(__file__)),
        "base_model": plan["base_model"], "base_model_revision": plan["base_model_revision"],
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "dataset_revision": plan["dataset_revision"],
        "train_sha256": plan["train_sha256"], "dev_sha256": plan["dev_sha256"],
        "train_label_counts": dict(counts),
        "class_weights": {label: float(weights[i]) for i, label in enumerate(LABELS)},
        "epoch_training_losses": epoch_losses, "training_seconds": training_seconds,
        "selected_run_directory": str(artifact_dir.relative_to(ROOT)),
        "model_sha256": sha256(model_file), "model_bytes": model_file.stat().st_size,
        "development_metrics": metrics,
        "prior_balanced_development_macro_f1": prior["selected_development_metrics"]["macro_f1"],
        "development_macro_f1_delta_vs_prior": (
            metrics["macro_f1"] - prior["selected_development_metrics"]["macro_f1"]),
        "original_test_result_known_before_experiment": True,
        "original_test_split_loaded_for_this_study": False,
        "claim_policy": plan["claim_policy"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.plan, args.report)
    print(json.dumps({"development_macro_f1": report["development_metrics"]["macro_f1"],
                      "development_macro_f1_delta_vs_prior": (
                          report["development_macro_f1_delta_vs_prior"])}, indent=2))


if __name__ == "__main__":
    main()
