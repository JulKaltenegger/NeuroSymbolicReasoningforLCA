
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.rand(1000, 1000, device="cuda")
    print("Tensor device:", x.device)

    
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda"
)
start = time.time()

embeddings = model.encode(texts)

print("Time:", time.time() - start)
