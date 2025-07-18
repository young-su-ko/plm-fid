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
# please have torch>=2.0 installed!
pip install plm-fid

#or with uv (highly recommended)
uv add plm-fid
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
|`model-name`            | 	The protein language model to use. Please specify a lowercase string, such as `esm2_8m`, `protbert`, or `antiberta2_cssp`. With API, see available models with `FrechetProteinDistance.available_models()`. For CLI use `--help`. Defaults to `esm2_650m`.|
|`device`           |	The device to run the model on, e.g., `cuda:0` or `cpu`. Defaults to cuda if available, otherwise `cpu`.|
|`max-length`       | Maximum length for each protein sequence. Longer sequences are truncated according to the selected truncation style. Some models may require a smaller max length (e.g., `antiberta2_cssp` supports up to 254). Defaults to `1000`.|
|`truncation-style` | How to truncate sequences longer than max-length. Use `end` to truncate from the back, or `center` to remove from the center, preserving N- and C-termini. Defaults to `center`.|
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
- Computing the FID will take longer for models with bigger dimensions (e.g. ESM2 3B) because of the larger covariance matrix.
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
> [!IMPORTANT]
> If you're using a standard protein language model with paired-chain data, note that the `|` character may be treated as an unknown token by most tokenizers. This typically doesn't cause a crash, but it can affect embedding quality.

## Examples
Here are two example use cases of pLM-based FID. These were primarily sanity-checks for myself to make sure the metric behaves as expected.
The central idea in both examples is the following:
Given two known protein distributions, $P$ and $Q$, that are believed to be distinct:
- Sample a **reference** set and set $A$ from distribution $P$
- Sample set $B$ from distribution $Q$.

then our hypothesis is:
```math
\texttt{plmFID}(\text{ref}, A) < \texttt{plmFID}(\text{ref}, B)
```
where $\texttt{plmFID}$ refers to the Fréchet distance computed using the mean and covariance of their pLM embeddings.

### Distinguishing CATH Classes
Using the [CATH S20 dataset](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/), this example tests whether pLM-based Fréchet distance can distinguish between CATH classes, specifically Class 1 and Class 2 proteins. Class 1 proteins are primarily **alpha-helical**, while Class 2 are mostly **beta-sheet**. Even though the Class is based on secondary-structure, we expect pLM embeddings to encode features that reflect these differences in fold.

Let $P$ be the distribution of Class 1 proteins, and $Q$ the distribution of Class 2 proteins.

We provide the FASTA files in `examples/cath` containing 1953 sequences each.
- `reference.fasta`: from $P$
- `class1.fasta`: set $A$ from $P$
- `class2.fasta`: set $B$ from $Q$

#### Results
| Model | plmFID(ref, $A$) | plmFID(ref, $B$) |
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

Across all pLMs, the expected behavior is observed. However, maybe this task is simply too trivial, so we move on to the second example. 

### Distinguishing antibodies binding to the SARS-CoV-2 Spike Protein's RBD vs NTD
The SARS-CoV-2 spike (S) protein contains two major domains that are frequent targets of neutralizing antibodies: the receptor-binding domain (RBD) and the N-terminal domain (NTD). Although both domains are part of the same protein, they differ in sequence and structure, raising the question of whether pLM embeddings can capture these biologically meaningful distinctions. In principle, antibodies that bind to the RBD versus the NTD should differ in their sequence features, as each domain presents distinct epitopes that shape antibody binding preferences.

From the [CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/) database, I filtered for antibodies with both heavy and light chains that bind to either the NTD or RBD. There were only 587 NTD antibodies compared to 7141 RBD antibodies, so to get equal number of sequences, I randomly sampled 587 RBD antibodies to be the reference and another distinct 587 RBD antibodies to be $A$.

Let $P$ be the distribution of RBD-binding antibodies, and $Q$ the distribution of NTD-binding antibodies.

We provide the FASTA files in `examples/cov` containing 587 sequences each.
- `reference.fasta`: from $P$
- `rbd.fasta`: set $A$ from $P$
- `ntd.fasta`: set $B$ from $Q$

#### Results
| Model | plmFID(ref, $A$) | plmFID(ref, $B$) |
| --- | --- | --- |
| AntiBERTa2-CSSP | 3.45 | 7.30 |

As expected, the distance between the reference and NTD antibodies is greater than the distance between the reference and RBD antibodies. 

### Conclusion
These initial sanity checks suggest that pLM-based FID can distinguish between protein groups with known structural and functional differences. That said, this was primarily a personal validation, a way to ensure the metric and code behave as expected. I’m not claiming this is a novel, comprehensive, or sufficient benchmark!

## Acknowledgements

### FID

Here's the paper that introduced FID: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500). I also wrote a short memo about FID and it's predecessor Inception Score [here](https://young-su-ko.github.io/notebook/fid.html).

### Existing works using Fréchet pLM Distance

I'd like to make it clear that pLM-based FID is not a **new idea**--here are the papers I've seen mention the use of pLM-based FID for assessing protein generations[^1].

- [Protein generation with evolutionary diffusion: sequence is all you need](https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1), Alamdari et al., 2023
- [Sampling Protein Language Models for Functional Protein Design](https://www.mlsb.io/papers_2023/Sampling_Protein_Language_Models_for_Functional_Protein_Design.pdf), Darmawan et al., 2023
- [Diffusion on language model encodings for protein sequence generation](https://arxiv.org/abs/2403.03726), Meshchaninov et al., 2024
- [Assessing Generative Model Coverage of Protein Structures with SHAPES](https://www.biorxiv.org/content/10.1101/2025.01.09.632260v2.full), Lu et al., 2025
- [ProtFlow: Fast Protein Sequence Design via Flow Matching on Compressed Protein Language Model Embeddings](https://arxiv.org/abs/2504.10983v1), Kong et al., 2025
- [Protein FID: Improved Evaluation of Protein Structure Generative Models](https://arxiv.org/abs/2505.08041), Faltings et al., 2025.

### Code

- Fréchet distance calculation implementation by [Dougal J. Sutherland](https://github.com/bioinf-jku/TTUR/blob/master/fid.py)
- `center` trunctation idea comes from [Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models](https://proceedings.mlr.press/v235/li24a.html):
    > "targeting signals often occur at the N- or C- terminal, and we reason that taking both terminals preserves biologically-relevant signals."

### Protein language models

| Model | Team | Link   | Paper |
| --- | --- | --- | --- |
| ESM2 | FAIR  | [Github](https://github.com/facebookresearch/esm)  | [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/10.1126/science.ade2574) |
| ProtBert (BFD) | RostLab | [HuggingFace](https://huggingface.co/Rostlab/prot_bert) | [ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning](https://ieeexplore.ieee.org/document/9477085) |
| ProtT5 | RostLab | [HuggingFace](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) |[ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning](https://ieeexplore.ieee.org/document/9477085) |
| ESMplusplus (ESMC) | EvolutionaryScale/Synthra | [HuggingFace](https://huggingface.co/Synthyra/ESMplusplus_small)/[Github](https://github.com/evolutionaryscale/esm) | - |
| AntiBERTa2-CSSP | Alchemab | [HuggingFace](https://huggingface.co/alchemab/antiberta2-cssp) | [Enhancing Antibody Language Models with Structural Information](https://www.mlsb.io/papers_2023/Enhancing_Antibody_Language_Models_with_Structural_Information.pdf) |


## Footnotes
[^1]: I'll be updating the list as I see more examples, as well as for any that I've missed.