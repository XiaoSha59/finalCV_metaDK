"""Table 3b fill / KD grid. Writes logs/audiovision_extend.csv and leaves locked checkpoints alone."""

import csv
import os
import time

from audiovision.process.train import train_metakd_audiovision

CKPT_DIR = os.path.join("audiovision", "checkpoints", "extend")
CSV_PATH = os.path.join("logs", "audiovision_extend.csv")
FIELDS = ["name", "missing", "rate", "fill", "kd_weight", "kd_space", "test_acc", "seconds", "ckpt"]


def _done():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def run_one(name, missing, rate, fill, kd_weight, kd_space="feature"):
    done = _done()
    if name in done:
        acc = float(done[name]["test_acc"])
        print(f"skip {name} test={acc:.2f}", flush=True)
        return acc

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    started = time.time()
    acc, ckpt = train_metakd_audiovision(
        missing_modality=missing,
        available_rate=rate,
        kd_weight=kd_weight,
        fill=fill,
        kd_space=kd_space,
        ckpt_dir=CKPT_DIR,
        ckpt_name=f"{name}.pth",
        verbose=True,
        epochs=60,
        batch_size=32,
        seed=42,
    )
    seconds = time.time() - started
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "missing": missing,
            "rate": rate,
            "fill": fill,
            "kd_weight": kd_weight,
            "kd_space": kd_space,
            "test_acc": f"{acc:.4f}",
            "seconds": f"{seconds:.1f}",
            "ckpt": ckpt,
        })
    print(f"DONE {name} test={acc:.2f} minutes={seconds / 60:.1f}", flush=True)
    return acc


def main():
    control = run_one("lower_star", "visual", 0.0, "copy", 0.0)
    fills = {}
    for fill in ("copy", "zero", "train_mean"):
        fills[fill] = run_one(f"b10_{fill}_a01", "visual", 0.10, fill, 0.1)
    winner = max(fills, key=fills.get)
    print(f"winner_fill={winner} acc={fills[winner]:.2f}", flush=True)

    alpha0 = run_one(f"b10_{winner}_a00", "visual", 0.10, winner, 0.0)
    kd_weight = 0.1 if fills[winner] >= alpha0 else 0.0
    print(f"winner_alpha={kd_weight} alpha0={alpha0:.2f} alpha01={fills[winner]:.2f}", flush=True)

    tag_alpha = f"a{int(round(kd_weight * 10)):02d}"
    rate_acc = {0.10: fills[winner] if kd_weight == 0.1 else alpha0}
    for rate in (0.05, 0.15, 0.20):
        tag = int(rate * 100)
        rate_acc[rate] = run_one(f"b{tag}_{winner}_{tag_alpha}", "visual", rate, winner, kd_weight)

    mean_b = sum(rate_acc[rate] for rate in (0.05, 0.10, 0.15, 0.20)) / 4.0
    image_10 = run_one(f"a10_{winner}_{tag_alpha}", "audio", 0.10, winner, kd_weight)

    logit = {}
    if mean_b <= control:
        print(f"feature mean {mean_b:.2f} <= LowerB* {control:.2f}; running logit KD", flush=True)
        for rate in (0.05, 0.10, 0.15, 0.20):
            tag = int(rate * 100)
            logit[rate] = run_one(f"blogit{tag}_{winner}", "visual", rate, winner, 0.1, kd_space="logit")

    print("EXTEND_SUMMARY", flush=True)
    print(f"LowerB* {control:.2f}", flush=True)
    print("fills10 " + " ".join(f"{name}={acc:.2f}" for name, acc in fills.items()), flush=True)
    print(
        "rates "
        + " ".join(f"{int(rate * 100)}={rate_acc[rate]:.2f}" for rate in (0.05, 0.10, 0.15, 0.20))
        + f" mean={mean_b:.2f} fill={winner} alpha={kd_weight}",
        flush=True,
    )
    print(f"3a_10 {image_10:.2f}", flush=True)
    if logit:
        logit_mean = sum(logit.values()) / 4.0
        print(
            "logit "
            + " ".join(f"{int(rate * 100)}={logit[rate]:.2f}" for rate in (0.05, 0.10, 0.15, 0.20))
            + f" mean={logit_mean:.2f}",
            flush=True,
        )
    print("EXTEND_DONE", flush=True)


if __name__ == "__main__":
    main()
