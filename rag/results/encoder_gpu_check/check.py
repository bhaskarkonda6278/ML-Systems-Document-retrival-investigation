import faulthandler
faulthandler.enable()

import sys
from pathlib import Path
sys.path.insert(0, str(Path("rag").resolve()))

print("Importing PyTorch", flush=True)
import torch
print("PyTorch:", torch.__version__, flush=True)
print("CUDA build:", torch.version.cuda, flush=True)
print("GPU available:", torch.cuda.is_available(), flush=True)

from src.encoder import LocalEncoder
print("Loading query encoder on GPU", flush=True)
encoder = LocalEncoder(Path("rag/models/enc_a"), device="cuda")

vector = encoder.encode("What affects a stock's opening price?")
print("Encoder test passed. Vector shape:", vector.shape, flush=True)
