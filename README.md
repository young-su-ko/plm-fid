# plm-fid: Fréchet Distance from pLM Embeddings
![cli_demo](https://raw.githubusercontent.com/young-su-ko/plm-fid/main/_assets/cli_demo.gif)

This tool computes the [Fréchet Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) between two sets of protein sequences based on their protein language model (pLM) embeddings. This metric is often used for image generation ([see original paper](https://arxiv.org/pdf/1706.08500)), but has also seen some popularity for protein generation ([see Acknowledgements](#Acknowledgements)).

When you pass in FASTA files, the specified model is loaded from the HuggingFace Hub using `transformers`. 
> [!NOTE]  
> If it's your first time using a given model, the weights will be downloaded and cached locally (this can take some time, especially for large models).

Each protein sequence is then embedded using the selected pLM. We apply mean pooling across the sequence dimension to obtain a fixed-size vector for each protein. After embedding both sets, we compute the mean vector and covariance matrix of each set. These summary statistics are then used to calculate the Fréchet Distance.

## Installation
> [!IMPORTANT]  
> Python >= 3.10 required. `sentencepiece` will give some issues with Python 3.13, so I recommend any version >=3.10, <3.13.
```bash
# if you have torch>=2.0 already installed
pip install plm-fid

# else
pip install plm-fid[torch]
```

## Usage

### CLI
```bash
plm-fid setA.fasta setB.fasta
```
> [!TIP]
> CLI expects paths to files only. These should be either `.fasta` files (raw protein sequences) or `.npy` or `.pt` files (pre-computed embeddings)

> [!TIP]
> To see all available options, run:
> ```bash
> plm-fid --help
> ```

| CLI/API Arguments | Description |
| --- | --- |
|`model-name`            | 	The protein language model to use. Please specify a lowercase string, such as `esm2_8m`, `protbert`, or `antiberta2_cssp`. See available models with `FrechetProteinDistance.available_models()`. Defaults to `esm2_650m`.|
|`device`           |	The device to run the model on, e.g., `cuda:0` or `cpu`. Defaults to cuda if available, otherwise `cpu`.|
|`max-length`       | Maximum length for each protein sequence. Longer sequences are truncated according to the selected truncation style. Some models may require a smaller max length (e.g., `antiberta2_cssp` supports up to 254). Defaults to `1000`.|
|`truncation-style` | How to truncate sequences longer than max-length. Use `end` to truncate from the back, or `center` to keep the central region. Defaults to `center`.|
|`batch-size`       | Number of sequences to embed per batch. Adjust according to available memory. Defaults to `1`.| 
|`save-embeddings`  | Whether to save the computed embeddings to `.npy` files. Useful for reuse or debugging. Disabled by default. |
|`output-dir`       | Directory to save output files if `--save-embeddings` is enabled. Defaults to current directory (`.`). |   


| CLI Only Arguments | Description |
| --- | --- |
|`round`          | Number of decimal places to round the final Fréchet distance result to. Defaults to `2`. |   
|`verbose`          | Show progress messages. Disabled by default. |   


### Python API

```python
from plm_fid import FrechetProteinDistance
import numpy as np
import torch

fid = FrechetProteinDistance(model_name="esmplusplus_small")

# Using saved NumPy arrays or PyTorch tensors
emb_a = np.load("embeddings_a.npy")       # shape: [N, D]
emb_b = torch.load("embeddings_b.pt")     # shape: [N, D]

distance = fid.compute_fid(emb_a, emb_b)
```
> [!NOTE]
> The API accepts both **file paths and in-memory arrays/tensors**. Argument names use underscores instead of dashes (e.g., model_name).

### Automatic Format Resolution in API
The `compute_fid()` method accepts:
- `.fasta` file paths
- `.npy` or `.pt` file paths
- In-memory NumPy arrays or PyTorch tensors

Any combination of the above are accepted by `compute_fid`.
```python
# FASTA
fid.compute_fid("set_a.fasta", "set_b.fasta")

# Embeddings from disk
fid.compute_fid("emb_a.npy", "emb_b.pt")

# Mixed input
import numpy as np
set_a = np.random.randn(10, 1280)
fid.compute_fid(set_a "emb_b.pt")
```

> [!IMPORTANT] 
> When using pre-computed embeddings, a stacked array/tensor with shape [batch, plm dim] is expected.


### **Important Notes**

- When mixing FASTA with pre-computed embeddings, make sure the model used to embed the FASTA file is the same as the one used to generate the `.npy` or `.pt` embeddings. A warning will be issued if there’s a mismatch.
    - If the embedding dimensions differ, `calculate_frechet_distance()` in `distance.py` will raise an error.
> [!WARNING]  
> However, if **different models produce embeddings of the same dimension**, this will not raise an error, but the FID is likely meaningless.

### **AntiBERTa2-Specific Notes**
- AntiBERTa2 has a max sequence length of 254. This will be enforced automatically.
- For paired-chain FASTA input, format each entry as:
```
>name
heavy_sequence|light_sequence
```

## Examples
### Distinguishing CATH Classes
We use the [CATH S20 dataset](https://www.google.com/placeholderfornow) to compare Class 1 and Class 2 proteins. Class 1 proteins are primarily **alpha-helical**, while Class 2 are mostly **beta-sheet**.

We provide three FASTA files in `examples/`
- `class1.fasta`: Class 1 proteins (1953 sequences)
- `class2.fasta`: Class 2 proteins (1953 sequences)
- `reference.fasta`: A different subset of Class 1 proteins, also 1953 sequences

Our hypothesis is that:
```math
\texttt{plmFID}(\text{ref, class1}) < \texttt{plmFID}(\text{ref, class2})
```
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
 

<!-- ### Distinguishing some binding antibodies (?)

#### Results
| Model | plmFID(ref, class1) | plmFID(ref, class2) |
| --- | --- | --- |
| AntiBERTa2-CSSP | X | X | -->

## Acknowledgements

## License
MIT 
