import torch
from Bio import SeqIO
from pathlib import Path
import warnings
from tqdm import tqdm
from transformers import logging as hf_logging

from .models import MODEL_MAP
from .model_configs import get_model_config


hf_logging.set_verbosity_error()


class ProteinEmbedder:
    """
    Internal class for embedding protein sequences using pretrained protein language models.
    This class is used by `FrechetProteinDistance` and is not intended for direct use (as of yet).
    """

    def __init__(
        self,
        model_name: str = "esm2_650m",
        device: str | None = None,
        max_length: int = 1000,
        truncation_style: str = "center",
        batch_size: int = 1,
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.truncation_style = truncation_style
        self.max_length = max_length
        self.model_path = MODEL_MAP[model_name]
        self.config = get_model_config(self.model_path)
        self.preprocessor = self.config.get("preprocessor")

        # Adjust max_length if required by model (e.g., antiberta2-cssp)
        model_max = self.config.get("max_sequence_length")
        if model_max and max_length > model_max:
            warnings.warn(f"Model max length is {model_max}. Truncating to that.")
            self.max_length = model_max

        self.model, self.tokenizer = self._load_model_and_tokenizer(model_name)
        self.model.eval()

    def _get_device(self, device):
        return device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model_and_tokenizer(self, model_name: str):
        model_instance = (
            self.config["model_class"]
            .from_pretrained(self.model_path, **self.config["model_kwargs"])
            .to(self.device)
        )
        # For now, only used by ESMplusplus models
        if self.config["tokenizer_on_model"]:
            tokenizer = model_instance.tokenizer
        else:
            tokenizer = self.config["tokenizer_class"].from_pretrained(
                self.model_path, **self.config["tokenizer_kwargs"]
            )
        return model_instance, tokenizer

    def _truncate_sequence(self, sequence: str) -> str:
        if len(sequence) <= self.max_length:
            return sequence

        if self.truncation_style == "end":
            return sequence[: self.max_length]

        else:  # preserve N- and C-termini, drop from the center
            keep_each_side = self.max_length // 2
            return sequence[:keep_each_side] + sequence[-keep_each_side:]

    @torch.no_grad()
    def _embed_batch(self, sequences: list[str]) -> torch.Tensor:
        """
        Embed a batch of sequences with mean pooling.
        """
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

    def _embed_fasta(self, fasta_path: str | Path) -> torch.Tensor:
        """
        Embed sequences from a FASTA file.
        """
        sequences = []
        path = Path(fasta_path)
        for record in SeqIO.parse(str(path), "fasta"):
            sequences.append(str(record.seq))

        if self.model_name == "antiberta2_cssp":
            # And no | exists in any fasta entry, raise warning that if using paired-chain fasta, each entry should be formatted as:
            # >name
            # heavy_sequence|light_sequence
            if not any("|" in seq for seq in sequences):
                warnings.warn(
                    "No | found in any fasta entry. If using paired-chain fasta, each entry should be formatted as: >name\nheavy_sequence|light_sequence"
                )

        all_embeddings = []
        for i in tqdm(
            range(0, len(sequences), self.batch_size), desc="Embedding proteins"
        ):
            batch_sequences = sequences[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_sequences)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)
