"""Live inference demo for the locked AudioVision MetaKD checkpoint.

Uses metakd_visual_rate_10pct: visual input was missing on 90% of training
pairs, and the saved test number is audio-only. The entropy gate is the
U-MetaKD rule (tau = 1). This checkpoint was trained as MetaKD, so the gate
is printed and is not applied to the prediction.
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.nn.functional as F

from audiovision.data.dataset import get_audiovision_loaders
from audiovision.process.models import MetaKDAudioVision

_CKPT = os.path.join(
    _REPO, "audiovision", "checkpoints", "metakd_visual_rate_10pct.pth"
)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def main():
    device = "cpu"
    torch.manual_seed(42)
    checkpoint = torch.load(_CKPT, map_location=device, weights_only=False)
    model = MetaKDAudioVision().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    weights = model.kd_weights().detach().cpu()

    _, _, test_loader = get_audiovision_loaders(
        batch_size=64,
        missing_modality="visual",
        available_rate=0.10,
        seed=42,
    )

    rows = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(audios=audios, eval_mode="audio_only")
            pred = logits.argmax(dim=-1)
            entropy = _entropy(logits)
            gate = torch.exp(-entropy / 1.0)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            for i in range(labels.size(0)):
                rows.append(
                    (
                        int(labels[i]),
                        int(pred[i]),
                        float(entropy[i]),
                        float(gate[i]),
                        bool(pred[i] == labels[i]),
                    )
                )

    rows.sort(key=lambda item: item[2])
    print("Checkpoint:", os.path.relpath(_CKPT, _REPO))
    print(
        "File record: "
        f"missing={checkpoint.get('missing_modality')}  "
        f"rate={checkpoint.get('available_rate')}  "
        f"test_acc={checkpoint.get('test_acc', float('nan')):.2f}%"
    )
    print(f"IWV softmax: visual {weights[0]:.3f}  audio {weights[1]:.3f}")
    print(f"Audio-only test accuracy now: {100.0 * correct / total:.2f}%  ({correct}/{total})")
    print()
    print("Lowest entropy (gate would keep these)")
    print(f"{'true':>6} {'pred':>6} {'H':>8} {'u':>8}  ok")
    for true, pred, entropy, gate, ok in rows[:5]:
        print(f"{true:6d} {pred:6d} {entropy:8.3f} {gate:8.3f}  {ok}")
    print()
    print("Highest entropy (gate would down-weight these)")
    for true, pred, entropy, gate, ok in rows[-5:]:
        print(f"{true:6d} {pred:6d} {entropy:8.3f} {gate:8.3f}  {ok}")


if __name__ == "__main__":
    main()
