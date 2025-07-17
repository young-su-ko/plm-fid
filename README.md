# pLM-FID: Protein Language Model Frechet Distance

Calculate Frechet Distance between sets of protein sequences using protein language model embeddings.

## Installation
```bash
# if you have torch>=2.0 already installed
pip install plm-fid

# else
pip install plm-fid[torch]
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

| API/CLI Arguments | Description |
| --- | --- |
|`model`            | 	The protein language model to use. Accepts either a PLM enum value or a lowercase string, such as `esm2_8m`, `protbert`, or `antiberta2_cssp`. Only used when input files are in FASTA format. Defaults to `esm2_650m`.|
|`device`           |	The device to run the model on, e.g., `cuda:0` or `cpu`. Defaults to cuda if available, otherwise `cpu`.|
|`max-length`       | Maximum length for each protein sequence. Longer sequences are truncated according to the selected truncation style. Some models may require a smaller max length (e.g., `antiberta2_cssp` supports up to 254). Defaults to `1000`.|
|`truncation-style` | How to truncate sequences longer than max-length. Use `end` to truncate from the back, or `center` to keep the central region. Defaults to `center`.|
|`batch-size`       | Number of sequences to embed per batch. Adjust according to available memory. Defaults to `1`.| 
|`save-embeddings`  | Whether to save the computed embeddings to `.npy` files. Useful for reuse or debugging. Disabled by default. |
|`output-dir`       | Directory to save output files if `--save-embeddings` is enabled. Defaults to current directory (`.`). |   

### CLI
```bash
plm-fid setA.fasta setB.fasta --model esm2_8m
```
| CLI Only Arguments | Description |
| --- | --- |
|`verbose`          | Show progress messages. Disabled by default. |   
> [!NOTE]
> Note: All other CLI options (`--model`, `--device`, etc.) map directly to the Python API arguments listed above.

### Automatic Format Resolution
The `compute_fid()` method accepts:
- `.fasta` file paths
- `.npy` or `.pt` file paths
- In-memory NumPy arrays or PyTorch tensors
```python
# FASTA
fid.compute_fid("set_a.fasta", "set_b.fasta")

# Embeddings from disk
fid.compute_fid("emb_a.npy", "emb_b.pt")

# Mixed input
fid.compute_fid("set_a.fasta", "emb_b.npy")
```

> [!IMPORTANT] 
> When using pre-computed embeddings, a stacked array/tensor with shape [batch, plm dim] is expected.


### **Important Notes**

- **When mixing FASTA with pre-computed embeddings**, ensure the model used at inference matches the one used to generate the `.npy` or `.pt` files. A warning will be shown otherwise.
    - If **model is different but dimensions are the same**, this will fail silently!
- AntiBERTa2 has a max sequence length of 254. This will be enforced automatically.

## Examples

We have two test cases in which pLM-based Fréchet distance should be able to meaningfully distinguish two different sets of protein sequences.

The fasta files for these examples are located in `example/cath` and `example/antibodies` respectively.

### Distinguishing CATH Classes
We use the [CATH S20 dataset](https://www.google.com/placeholderfornow) to compare Class 1 and Class 2 proteins. Class 1 proteins are primarily **alpha-helical**, while Class 2 are mostly **beta-sheet**.

We provide three FASTA files:
- `class1.fasta`: Class 1 proteins (1953 sequences)
- `class2.fasta`: Class 2 proteins (1953 sequences)
- `reference.fasta`: A different subset of Class 1 proteins, also 1953 sequences


Our hypothesis is that
$$
\texttt{plmFID}(\text{ref, class1}) < \texttt{plmFID}(\text{ref, class2})
$$

Even though CATH classification is structural, we expect pLM embeddings to encode features that reflect these differences in fold.

#### Results
| Model | plmFID(ref, class1) | plmFID(ref, class2) |
| --- | --- | --- |
| ESM2 (8M) | 0.11 | 2.52 |
| ESM2 (35M)| 0.16 | 2.15 |
| ESM2 (150M)| 0.15 | 1.70 |
| ESM2 (650M)| 0.38 | 2.18 |
| ESM2 (3B)| 1.07 | 3.86 |
| ESM2 (15B)| 7.94 | 18.17 |
| ProtBert | 0.08 | 0.49 |
| ProtBert BFD| 0.07 | 0.75 |
| ProtT5| 0.15 | 0.74 |
| ESM++ (S)| 0.01 | 0.07 |
| ESM++ (L)| 0.01 | 0.11 |
 

### Distinguishing some binding antibodies (?)

#### Results
| Model | plmFID(ref, class1) | plmFID(ref, class2) |
| --- | --- | --- |
| AntiBERTa2-CSSP | X | X |

## License
MIT 
