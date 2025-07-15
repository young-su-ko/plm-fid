import torch
from Bio import SeqIO
from pathlib import Path
import warnings
from tqdm import tqdm
import numpy as np
from transformers import logging as hf_logging

from .models import PLM
from .model_configs import get_model_config


hf_logging.set_verbosity_error()


class ProteinEmbedder:
    def __init__(
        self,
        model_name: str = PLM.ESM2_650M.value,
        device: str | None = None,
        max_length: int = 1000,
        truncation_style: str = "center",
        batch_size: int = 1,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        config = get_model_config(model_name)
        model_class = config["model_class"]
        tokenizer_class = config["tokenizer_class"]
        model_kwargs = config["model_kwargs"]
        tokenizer_kwargs = config["tokenizer_kwargs"]
        self.preprocessor = config["preprocessor"]

        # Check max_length against model constraints
        model_max_length = config.get("max_sequence_length")
        if model_max_length is not None and max_length > model_max_length:
            warnings.warn(
                f"Model '{model_name}' has a maximum sequence length of {model_max_length}. "
                f"Provided max_length={max_length} will be capped.",
                UserWarning,
            )
            max_length = model_max_length

        self.model = model_class.from_pretrained(model_name, **model_kwargs).to(device)
        self.model.eval()

        if config["tokenizer_on_model"]:
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, **tokenizer_kwargs
            )

        self.max_length = max_length
        assert truncation_style in ["end", "center"], (
            "truncation_style must be 'end' or 'center'"
        )
        self.truncation_style = truncation_style
        self.batch_size = batch_size

    def _truncate_sequence(self, sequence: str) -> str:
        if len(sequence) <= self.max_length:
            return sequence

        if self.truncation_style == "end":
            return sequence[: self.max_length]
        else:  # center
            start = (len(sequence) - self.max_length) // 2
            return sequence[start : start + self.max_length]

    @torch.no_grad()
    def _embed_batch(self, sequences: list[str]) -> torch.Tensor:
        sequences = [self._truncate_sequence(seq) for seq in sequences]

        if self.preprocessor:
            sequences = [self.preprocessor(seq) for seq in sequences]

        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(
            self.device
        )
        outputs = self.model(**inputs)

        # Mean pool over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu()

    def _embed_fasta(self, fasta_path: str | Path) -> np.ndarray:
        sequences = []
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            sequences.append(str(record.seq))

        all_embeddings = []
        for i in tqdm(
            range(0, len(sequences), self.batch_size), desc="Embedding proteins"
        ):
            batch_sequences = sequences[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_sequences)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def check_embedding_sizes(emb_a: np.ndarray, emb_b: np.ndarray):
        if (
            abs(len(emb_a) - len(emb_b)) > min(len(emb_a), len(emb_b)) * 0.1
        ):  # 10% difference
            warnings.warn(
                f"Embedding sets have significantly different sizes ({len(emb_a)} vs {len(emb_b)}). "
                "This may affect the reliability of the covariance estimation."
            )
