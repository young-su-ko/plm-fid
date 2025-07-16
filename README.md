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

distance = fid.compute_fid(emb_a, emb_b)
```

| Argument | Description |
| --- | --- |
|`model`            | Which protein language model to use for embedding. We currently provide 12 options: ESM2 family, ProtBert(BFD), ProtT5, ESM++(small/large), AntiBERTa2-CSSP. Defaults to ESM2_650M|
|`device`           | Device used to accelerate embedding generation. Will auto-detect if unspecified.|
|`max-length`       | Maximum length of protein sequences allowed before truncation. Defaults to 1000.|
|`truncation-style` | Either `end` or `center`. Defaults to `center`.|
|`batch-size`       | Batch size for embedding. Needs adjustment depending on available memory. Defaults to 1.| 
|`save-embeddings`  | Save embeddings to disk for reuse. Disabled by default. |
|`output-dir`       | If `--save-embeddings` embeddings will be saved in this directory. |   
|`verbose`          | Show progress messages. Defaults to False |   


### Automatic Format Resolution

The `compute_fid()` method accepts:
- `.fasta` file paths
- `.npy` or `.pt` file paths
- In-memory NumPy arrays or PyTorch tensors

> [!IMPORTANT] 
> When using pre-computed embeddings, a stacked array/tensor with shape [batch, plm dim] is expected.
```python
# FASTA
fid.compute_fid("set_a.fasta", "set_b.fasta")

# Embeddings from disk
fid.compute_fid("emb_a.npy", "emb_b.pt")

# Mixed input
fid.compute_fid("set_a.fasta", "emb_b.npy")
```

### CLI
```bash
plm-fid --help
```
```bash
plm-fid setA.fasta setB.fasta --model esm2_8m
```
All combinations of .fasta, .npy, or .pt files are supported for CLI. In Python, in-memory NumPy arrays and PyTorch tensors are also accepted.

### **Important Notes**

- **When mixing FASTA with pre-computed embeddings**, ensure the model used at inference matches the one used to generate the `.npy` or `.pt` files. A warning will be shown otherwise.
- AntiBERTa2 has a max sequence length of 254. This will be enforced automatically.

## Examples

We have two test cases in which pLM-based Fréchet distance should be able to meaningfully distinguish two different sets of protein sequences.

The fasta files for these examples are located in `example/cath` and `example/antibodies` respectively.

### Distinguishing CATH Classes
From the [CATH S20 dataset](www.google.com/placeholderfornow) we retrieved sequences corresponding to Class 1 and Class 2. As a reminder, Class 1 proteins are mostly alpha helices, while Class 2 proteins are mostly beta sheets. 

We provide three fasta files:

`class1.fasta` contains protein sequences from Class 1 while `class2.fasta` contains proteins sequences in Class 2. We construct a `reference.fasta` to only contain other Class 1 sequences not in `class1.fasta`. All three files have 1953 sequences.

Our hypothesis is that
$$
\texttt{plmFID}(\text{ref, class1}) < \texttt{plmFID}(\text{ref, class2})
$$

While the CATH classes are based on structure, the pLM embedding provides enough information such that sequences belonging to Class 1 are more similar to each other than to Class 2.

#### Results
| Model | plmFID(ref, class1) | plmFID(ref, class2) |
| --- | --- | --- |
| ESM2 (8M) | X | X |
| ESM2 (150M)| X | X |
| ESM2 (640M)| X | X |
| ESM2 (3B)| X | X |
| ESM2 (15B)| X | X |
| ProtBert| X | X |
| ProtBert BFD| X | X |
| ProtT5| X | X |
| ESM++ (S)| X | X |
| ESM++ (L)| X | X |
 

### Distinguishing some binding antibodies (?)

#### Results
| Model | plmFID(ref, class1) | plmFID(ref, class2) |
| --- | --- | --- |
| AntiBERTa2-CSSP | X | X |

## License
MIT 
