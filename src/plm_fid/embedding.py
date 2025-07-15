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
        model: PLM = PLM.ESM2_650M,
        device: str | None = None,
        max_length: int = 1000,
        truncation_style: str = "center",
        batch_size: int = 1,
    ):
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.truncation_style = truncation_style
        self.max_length = max_length

        config = get_model_config(str(model))
        self.preprocessor = config.get("preprocessor")

        # Adjust max_length if required by model (e.g., antiberta2-cssp)
        model_max = config.get("max_sequence_length")
        if model_max and max_length > model_max:
            warnings.warn(f"Model max length is {model_max}. Truncating to that.")
            self.max_length = model_max

        self.model, self.tokenizer = self._load_model_and_tokenizer(model, config)
        self.model.eval()

    def _get_device(self, device):
        return device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model_and_tokenizer(self, model, config):
        model_instance = (
            config["model_class"]
            .from_pretrained(str(model), **config["model_kwargs"])
            .to(self.device)
        )
        # For now, only used by ESMplusplus models
        if config["tokenizer_on_model"]:
            tokenizer = model_instance.tokenizer
        else:
            tokenizer = config["tokenizer_class"].from_pretrained(
                str(model), **config["tokenizer_kwargs"]
            )
        return model_instance, tokenizer

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
