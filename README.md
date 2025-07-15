# pLM-FID: Protein Language Model Frechet Distance

Calculate Frechet Distance between sets of protein sequences using protein language model embeddings.

## Installation

```bash
pip install plm-fid
```

## Usage

### Python API

```python
from plm_fid import FrechetProteinDistance
import numpy as np
import torch

fid = FrechetProteinDistance()

# Using raw NumPy arrays or PyTorch tensors
emb_a = np.load("embeddings_a.npy")       # shape: [N, D]
emb_b = torch.load("embeddings_b.pt")     # shape: [N, D]

score = fid.compute_fid(emb_a, emb_b)
```

### Automatic Format Resolution

The `compute_fid()` method accepts:
- `.fasta` file paths
- `.npy` or `.pt` embedding files
- In-memory NumPy arrays or PyTorch tensors
```python
# FASTA vs. FASTA
fid.compute_fid("set_a.fasta", "set_b.fasta")

# Embeddings from disk
fid.compute_fid("emb_a.npy", "emb_b.pt")

# Mixed input
fid.compute_fid("set_a.fasta", "emb_b.npy")
```

### CLI
*in progress*

### **Important Notes**
- When using pre-computed embeddings, a stacked array/tensor with shape [batch, plm dim] is expected.
- When mixing FASTA files with pre-computed embeddings, make sure the model you select matches the model used to generate the pre-computed embeddings.
- AntiBERTa-2 has a specific maximum sequence length of 254. If you specify a larger max_length, we will enforce this length and truncate based on your specified choice.

## Features
- Calculate Frechet distance between sets of protein sequences using pLM embeddings
- Support for both pre-computed embeddings and FASTA input
- Flexible input combinations (FASTA + FASTA, FASTA + pre-computed, or pre-computed + pre-computed)
- Default GPU acceleration with fallback to CPU
- Configurable sequence length truncation
- Save embeddings as .npy for reuse
- Flexible batch size for memory management
- Support for models available on Huggingface

## License

MIT 