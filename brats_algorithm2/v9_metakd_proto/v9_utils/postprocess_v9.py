import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "brats"))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from postprocess_brats import postprocess_prediction, postprocess_batch
except ImportError:
    try:
        from process.postprocess_brats import postprocess_prediction, postprocess_batch
    except ImportError:
        def postprocess_prediction(pred_binary, et_thresh=200):
            return pred_binary
        def postprocess_batch(pred_batch, et_thresh=200):
            return pred_batch

__all__ = ["postprocess_prediction", "postprocess_batch"]
