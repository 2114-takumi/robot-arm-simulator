"""Fine-tune R1 on BC + R1 + R2 with capped hard-example weighted loss."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vision_dagger_common import PROJECT_ROOT, PYTORCH_ROOT, load_policy, preprocess_rgb, sha256

NORMALIZATION_KEYS = ("proprio_mean", "proprio_std", "tau_mean", "tau_std")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_source(root, dagger):
    """Student predictions determine weights only. Targets are always tau1,tau2."""
    csv_path = root / "labels.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty dataset: {root}")
    images = np.empty((len(rows), 3, 60, 80), dtype=np.uint8)
    states = np.empty((len(rows), 4), dtype=np.float32)
    targets = np.empty((len(rows), 2), dtype=np.float32)
    errors = np.zeros(len(rows), dtype=np.float64)
    episodes = np.empty(len(rows), dtype=np.int64)
    digest = hashlib.sha256(csv_path.read_bytes())
    for i, row in enumerate(rows):
        episodes[i] = int(row["episode"])
        states[i] = [float(row[k]) for k in ("q1", "q2", "dq1", "dq2")]
        expert = np.array([float(row[k]) for k in ("tau1", "tau2")])
        targets[i] = expert
        if dagger:
            student = np.array([float(row[k]) for k in ("student_tau1", "student_tau2")])
            errors[i] = np.linalg.norm(student - expert)
            if "action_error" in row and not np.isclose(float(row["action_error"]), errors[i], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Inconsistent action_error in {csv_path}, row {i}")
        path = root / "images" / row["filename"]
        raw = path.read_bytes()
        digest.update(row["filename"].encode())
        digest.update(hashlib.sha256(raw).digest())
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable image: {path}")
        processed = preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # Shared preprocessing produces uint8 resize /255; cache its exact bytes.
        images[i] = np.rint(processed * 255.0).astype(np.uint8)
        if not np.allclose(images[i].astype(np.float32) / 255.0, processed, rtol=0, atol=1e-7):
            raise ValueError("Image cache would change preprocessing")
    if not all(np.isfinite(x).all() for x in (states, targets, errors)) or (episodes < 0).any():
        raise ValueError(f"Invalid states/labels in {csv_path}")
    return {"images": images, "states": states, "targets": targets,
            "errors": errors, "episodes": episodes,
            "provenance": {"root": str(root.resolve()), "samples": len(rows),
                           "labels_sha256": sha256(csv_path), "dataset_sha256": digest.hexdigest()}}


def hard_weights(errors, scale):
    return (1.0 + 2.0 * np.clip(np.asarray(errors) / max(scale, 1e-6), 0.0, 1.0)).astype(np.float32)


def weighted_mse(predictions, targets, weights):
    per_sample = (predictions - targets).square().mean(dim=1)
    return (per_sample * weights).sum() / weights.sum()


def original_split(episodes):
    train = np.asarray(episodes) < 80
    test = ~train
    if train.sum() != 4000 or test.sum() != 1000:
        raise ValueError("Expected Original BC episode<80:4000, episode>=80:1000")
    return train, test


def make_dataset(images, states, targets, weights, checkpoint):
    return TensorDataset(torch.from_numpy(images),
                         (torch.from_numpy(states) - checkpoint["proprio_mean"]) / checkpoint["proprio_std"],
                         (torch.from_numpy(targets) - checkpoint["tau_mean"]) / checkpoint["tau_std"],
                         torch.from_numpy(weights))


def offline_test(model, loader, device, checkpoint):
    model.eval()
    squared, count = np.zeros(2), 0
    with torch.no_grad():
        for images, states, targets, _ in loader:
            prediction = model(images.to(device, dtype=torch.float32) / 255.0, states.to(device))
            error = (prediction.cpu() - targets) * checkpoint["tau_std"]
            squared += error.double().square().sum(0).numpy()
            count += len(images)
    return {"samples": count, "action_rmse_nm": np.sqrt(squared / count).tolist(),
            "used_for_training_or_model_selection": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in [("bc-dir", PROJECT_ROOT / "outputs/step_13_b_1"),
                            ("r1-dir", PROJECT_ROOT / "outputs/step_13_b_5_dagger_round1"),
                            ("r2-dir", PROJECT_ROOT / "outputs/step_13_b_8_dagger_round2"),
                            ("source-checkpoint", PYTORCH_ROOT / "checkpoints/vision_dagger_r1_policy.pt"),
                            ("checkpoint", PYTORCH_ROOT / "checkpoints/vision_dagger_r2_policy.pt"),
                            ("output-dir", PROJECT_ROOT / "outputs/step_13_b_9_train_dagger_round2")]:
        parser.add_argument("--" + option, type=Path, default=default)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.source_checkpoint.resolve():
        parser.error("Destination must differ from the R1 checkpoint")
    if args.checkpoint.exists() or args.output_dir.exists():
        raise FileExistsError("Checkpoint and training output directory must both be new")
    if min(args.epochs, args.batch_size, args.threads) < 1 or not 0 < args.min_lr <= args.lr:
        parser.error("Require positive epochs/batch-size/threads and 0 < min-lr <= lr")
    manifest = json.loads((args.r2_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["student_sha256"] != sha256(args.source_checkpoint):
        raise ValueError("R2 must be complete and collected using this exact R1 checkpoint")
    if manifest["labels_sha256"] != sha256(args.r2_dir / "labels.csv"):
        raise ValueError("R2 labels changed since collection")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, checkpoint = load_policy(args.source_checkpoint)
    print(f"Preloading datasets; device={device}, threads={args.threads}", flush=True)
    bc, r1, r2 = read_source(args.bc_dir, False), read_source(args.r1_dir, True), read_source(args.r2_dir, True)
    train_mask, test_mask = original_split(bc["episodes"])
    if len(r1["errors"]) != 5000 or len(r2["errors"]) != manifest["samples"]:
        raise ValueError("Unexpected R1/R2 sample count")
    scale = max(float(np.percentile(np.r_[r1["errors"], r2["errors"]], 90)), 1e-6)
    weights1, weights2 = hard_weights(r1["errors"], scale), hard_weights(r2["errors"], scale)
    weights = np.r_[np.ones(4000, dtype=np.float32), weights1, weights2]
    arrays = [np.concatenate([bc[k][train_mask], r1[k], r2[k]]) for k in ("images", "states", "targets")]
    train = make_dataset(*arrays, weights, checkpoint)
    heldout = make_dataset(bc["images"][test_mask], bc["states"][test_mask], bc["targets"][test_mask],
                           np.ones(1000, dtype=np.float32), checkpoint)
    def weight_stats(w):
        return {"min": float(w.min()), "mean": float(w.mean()), "max": float(w.max()),
                "fraction_at_cap": float(np.mean(w == 3))}
    provenance = {
        "sample_counts": {"bc_train": 4000, "r1_train": len(weights1), "r2_train": len(weights2),
                          "total_train": len(train), "bc_heldout": 1000},
        "sources": {k: source["provenance"] for k, source in (("bc", bc), ("r1", r1), ("r2", r2))},
        "source_checkpoint": str(args.source_checkpoint.resolve()), "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "original_split": {"rule": "episode<80 train, episode>=80 heldout; unchanged from BC/R1",
                           "train_row_indices": np.flatnonzero(train_mask).tolist(),
                           "heldout_row_indices": np.flatnonzero(test_mask).tolist()},
        "normalization": {key: checkpoint[key].tolist() for key in NORMALIZATION_KEYS},
        "normalization_refitted": False,
        "hard_example_weighting": {"formula": "BC:1; DAgger:1+2*clip(action_error/p90,0,1)",
                                   "action_error": "norm(recorded student_tau - expert_tau) in Nm",
                                   "scale_nm": scale, "max_weight": 3.0,
                                   "r1": weight_stats(weights1), "r2": weight_stats(weights2)},
        "settings": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "device_used": str(device), "optimizer": "Adam", "scheduler": "cosine",
                     "gradient_clip_norm": 5.0, "torch_version": str(torch.__version__),
                     "model_selection": "fixed final epoch; no heldout or closed-loop selection"},
    }
    del bc, r1, r2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "provenance.json", provenance)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    test_loader = DataLoader(heldout, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    history, start = [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_weighted, total_weight, total_unweighted, seen = 0.0, 0.0, 0.0, 0
        epoch_start, lr = time.perf_counter(), optimizer.param_groups[0]["lr"]
        for images, states, targets, batch_weights in loader:
            images, states = images.to(device, dtype=torch.float32) / 255.0, states.to(device)
            targets, batch_weights = targets.to(device), batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images, states)
            per_sample = (predictions - targets).square().mean(dim=1)
            loss = weighted_mse(predictions, targets, batch_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_weighted += (per_sample.detach() * batch_weights).sum().item()
            total_weight += batch_weights.sum().item()
            total_unweighted += per_sample.detach().sum().item()
            seen += len(images)
        scheduler.step()
        record = {"epoch": epoch, "lr": lr, "weighted_mse": total_weighted / total_weight,
                  "unweighted_mse": total_unweighted / seen, "seconds": time.perf_counter() - epoch_start}
        history.append(record)
        write_json(args.output_dir / "training_trace.json", history)
        print(json.dumps(record), flush=True)
    # Held-out data are consulted once, only after the fixed training run.
    report = {**provenance, "training_trace": history, "training_seconds": time.perf_counter() - start,
              "heldout_offline": offline_test(model, test_loader, device, checkpoint),
              "checkpoint": str(args.checkpoint.resolve())}
    result = {**checkpoint, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
              "training_metadata": report}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("xb") as stream:
        torch.save(result, stream)
    report["checkpoint_sha256"] = sha256(args.checkpoint)
    write_json(args.output_dir / "summary.json", report)
    print(json.dumps({"sample_counts": report["sample_counts"], "heldout_offline": report["heldout_offline"],
                      "checkpoint": report["checkpoint"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
