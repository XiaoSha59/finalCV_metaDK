import os
import sys
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALG2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BRATS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "brats"))
for p in [CURRENT_DIR, ALG2_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v8_models.dual_net_v8 import DualNet_V8
from v8_losses.loss_v8 import PromptDiversityRegularizer


def test_v8_forward_and_loss():
    print("=== Testing V8: MetaKD-Prompt Local Smoke Test ===")
    B, C, D, H, W = 1, 4, 32, 32, 32
    images = torch.randn(B, C, D, H, W)

    model = DualNet_V8(num_classes=3)
    model.train()

    # 1. Forward missing mode (Student: 0, 1 -> missing 2: T1ce, 3: T2)
    preds, mode_split, kd_loss = model(images, mode="0,1")
    print(f"[OK] Prompt-imputed forward: shape={preds.shape}, modes={mode_split}")

    # 2. Test Prompt Diversity Regularizer
    reg = PromptDiversityRegularizer()
    loss_ortho = reg(model.modality_prompts)
    print(f"[OK] Prompt Ortho Loss: {loss_ortho.item():.4f}")

    # 3. Test Backward pass on prompt parameters
    total_loss = preds.sum() + loss_ortho
    total_loss.backward()
    assert model.modality_prompts.grad is not None, "Prompt grad should not be None"
    assert model.prompt_alpha.grad is not None, "Alpha grad should not be None"
    print(f"[OK] Prompt grad norm: {model.modality_prompts.grad.norm().item():.4f}")
    print(f"[OK] Alpha grad: {model.prompt_alpha.grad.item():.4f}")
    print("=== All V8 Tests PASSED! ===")


if __name__ == "__main__":
    test_v8_forward_and_loss()
