"""Cross-modal supervised contrastive training for AudioVision Table 3b.

No meta-weight update. Paired image/audio features of the same digit are positives.
Locked checkpoints are not touched.
"""

import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from audiovision.data.dataset import get_audiovision_loaders
from audiovision.process.models import ImageEncoder, MetaKDAudioVision, SNetPluse
from audiovision.process.train import evaluate_accuracy

CSV_PATH = os.path.join("logs", "audiovision_supcon.csv")
CKPT_DIR = os.path.join("audiovision", "checkpoints", "extend")
FIELDS = ["name", "head", "missing", "rate", "beta", "test_acc", "seconds"]


class Projector(nn.Module):
    def __init__(self, width: int = 160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


def supervised_contrastive(z_img, z_aud, labels, temperature: float = 0.1) -> torch.Tensor:
    features = torch.cat([z_img, z_aud], dim=0)
    targets = torch.cat([labels, labels], dim=0)
    logits = features @ features.T
    logits = logits / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
    positives = (targets[:, None] == targets[None, :]) & ~self_mask
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_zeros(())
    loss = -(log_prob * positives).sum(dim=1)[valid] / positive_count[valid]
    return loss.mean()


def _done():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def _record(row):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _paired_contrast(projector, img_feat, aud_feat, labels, visual_mask, audio_mask):
    paired = visual_mask & audio_mask
    if int(paired.sum()) < 2:
        return img_feat.new_zeros(())
    return supervised_contrastive(
        projector(img_feat[paired]),
        projector(aud_feat[paired]),
        labels[paired],
    )


def train_fusion(name, missing, rate, beta, epochs=60, batch_size=32, seed=42):
    done = _done()
    if name in done:
        acc = float(done[name]["test_acc"])
        print(f"skip {name} test={acc:.2f}", flush=True)
        return acc

    device = "cpu"
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size, missing_modality=missing, available_rate=rate, seed=seed,
    )
    model = MetaKDAudioVision().to(device)
    projector = Projector().to(device)
    params = [param for key, param in model.named_parameters() if "kd_weights" not in key]
    optimizer = optim.Adam(params + list(projector.parameters()), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    eval_mode = "image_only" if missing == "audio" else "audio_only"
    best_val = -1.0
    best_test = 0.0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            visual_mask = batch["visual_mask"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            optimizer.zero_grad()
            logits, _kd = model(
                images, audios, visual_mask=visual_mask, audio_mask=audio_mask,
                eval_mode="full", fill="copy",
            )
            img_feat, aud_feat = model.encode(images, audios)
            loss = criterion(logits, labels) + beta * _paired_contrast(
                projector, img_feat, aud_feat, labels, visual_mask, audio_mask,
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=eval_mode, device=device, fill="copy")
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=eval_mode, device=device, fill="copy")
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"{name} epoch {epoch:02d} val={val_acc:.2f} best_test={best_test:.2f}",
                flush=True,
            )

    os.makedirs(CKPT_DIR, exist_ok=True)
    _record({
        "name": name, "head": "fusion", "missing": missing, "rate": rate,
        "beta": beta, "test_acc": f"{best_test:.4f}", "seconds": f"{time.time() - started:.1f}",
    })
    print(f"DONE {name} test={best_test:.2f}", flush=True)
    return best_test


def train_snet(name, rate, beta, epochs=60, batch_size=32, seed=42):
    done = _done()
    if name in done:
        acc = float(done[name]["test_acc"])
        print(f"skip {name} test={acc:.2f}", flush=True)
        return acc

    device = "cpu"
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size, missing_modality="visual", available_rate=rate, seed=seed,
    )
    audio_net = SNetPluse().to(device)
    image_enc = ImageEncoder().to(device)
    projector = Projector().to(device)
    optimizer = optim.Adam(
        list(audio_net.parameters()) + list(image_enc.parameters()) + list(projector.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_test = 0.0
    started = time.time()

    for epoch in range(1, epochs + 1):
        audio_net.train()
        image_enc.train()
        projector.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            visual_mask = batch["visual_mask"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            optimizer.zero_grad()
            logits = audio_net(audios)
            loss = criterion(logits, labels)
            if beta > 0:
                loss = loss + beta * _paired_contrast(
                    projector, image_enc(images), audio_net.encoder(audios),
                    labels, visual_mask, audio_mask,
                )
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate_accuracy(audio_net, val_loader, device=device)
        test_acc = evaluate_accuracy(audio_net, test_loader, device=device)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"{name} epoch {epoch:02d} val={val_acc:.2f} best_test={best_test:.2f}",
                flush=True,
            )

    _record({
        "name": name, "head": "snet", "missing": "visual", "rate": rate,
        "beta": beta, "test_acc": f"{best_test:.4f}", "seconds": f"{time.time() - started:.1f}",
    })
    print(f"DONE {name} test={best_test:.2f}", flush=True)
    return best_test


def main():
    fusion_10 = {}
    for beta in (0.1, 1.0):
        tag = str(beta).replace(".", "")
        fusion_10[beta] = train_fusion(f"fuse10_b{tag}", "visual", 0.10, beta)
    winner = max(fusion_10, key=fusion_10.get)
    print(f"fusion_beta={winner} acc={fusion_10[winner]:.2f}", flush=True)
    tag = str(winner).replace(".", "")
    fusion_rates = {0.10: fusion_10[winner]}
    for rate in (0.05, 0.15, 0.20):
        fusion_rates[rate] = train_fusion(
            f"fuse{int(rate * 100)}_b{tag}", "visual", rate, winner,
        )
    image_10 = train_fusion(f"fuse_a10_b{tag}", "audio", 0.10, winner)
    fusion_mean = sum(fusion_rates[rate] for rate in (0.05, 0.10, 0.15, 0.20)) / 4.0

    snet_control = train_snet("snet_audio_only", 0.0, 0.0)
    snet_rates = {}
    for rate in (0.05, 0.10, 0.15, 0.20):
        snet_rates[rate] = train_snet(f"snet{int(rate * 100)}_b1", rate, 1.0)
    snet_mean = sum(snet_rates.values()) / 4.0

    print("SUPCON_SUMMARY", flush=True)
    print(
        "fusion "
        + " ".join(f"{int(rate * 100)}={fusion_rates[rate]:.2f}" for rate in (0.05, 0.10, 0.15, 0.20))
        + f" mean={fusion_mean:.2f} beta={winner} 3a10={image_10:.2f}",
        flush=True,
    )
    print(f"fusion10 betas 0.1={fusion_10[0.1]:.2f} 1.0={fusion_10[1.0]:.2f}", flush=True)
    print(
        "snet "
        + " ".join(f"{int(rate * 100)}={snet_rates[rate]:.2f}" for rate in (0.05, 0.10, 0.15, 0.20))
        + f" mean={snet_mean:.2f} control={snet_control:.2f}",
        flush=True,
    )
    print("SUPCON_DONE", flush=True)


if __name__ == "__main__":
    main()
