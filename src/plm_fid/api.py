from .utils import (
    get_mu_sigma,
    resolve_input_to_numpy,
)
from .embedding import ProteinEmbedder
from .distance import calculate_frechet_distance
from .models import PLM
from pathlib import Path
import warnings
import numpy as np
import torch
import logging


class FrechetProteinDistance:
    def __init__(
        self,
        model: PLM = PLM.ESM2_650M,
        device: str | None = None,
        max_length: int = 1000,
        truncation_style: str = "end",
        batch_size: int = 1,
        save_embeddings: bool = False,
        output_dir: str | Path = ".",
    ):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.truncation_style = truncation_style
        self.batch_size = batch_size
        self.save_embeddings = save_embeddings
        self.output_dir = output_dir

        self.embedder = ProteinEmbedder(
            model=self.model,
            device=self.device,
            max_length=self.max_length,
            truncation_style=self.truncation_style,
            batch_size=self.batch_size,
        )

    def compute_fid(
        self,
        set_a: str | Path | np.ndarray | torch.Tensor,
        set_b: str | Path | np.ndarray | torch.Tensor,
    ) -> float:
        """
        Compute the Fréchet distance between two sets of protein representations.

        Accepts either FASTA files, precomputed embedding files (.npy or .pt), or
        raw embedding arrays/tensors and returns the Fréchet distance between the two sets.

        Parameters
        ----------
        set_a : str or Path or np.ndarray or torch.Tensor
            The first protein set. Can be:
            - Path to a `.fasta` file
            - Path to a `.npy` or `.pt` file containing precomputed embeddings
            - A NumPy array of shape `(N, D)`
            - A PyTorch tensor of shape `(N, D)`

        set_b : str or Path or np.ndarray or torch.Tensor
            The second protein set. Same supported formats as `set_a`.

        Returns
        -------
        float
            The Fréchet distance between the embeddings of two protein sets.

        Raises
        ------
        ValueError
            If either input cannot be resolved into a valid 2D embedding array.

        TypeError
            If the input types are unsupported.

        Warnings
        --------
        UserWarning
            If `set_a` and `set_b` are in different formats (e.g., one FASTA, one .npy),
            a warning is issued to ensure consistent protein language model usage.

        Examples
        --------
        >>> fid = FrechetProteinDistance()
        >>> fid.compute_fid("set_a.fasta", "set_b.fasta")
        2.54

        >>> fid.compute_fid("set_a.npy", "set_b.npy")
        0.91

        >>> import numpy as np
        >>> a = np.random.randn(10, 256)
        >>> b = np.random.randn(10, 256)
        >>> fid.compute_fid(a, b)
        1.12
        """
        # Warn if set types are mismatched
        if (
            isinstance(set_a, (str, Path))
            and isinstance(set_b, (str, Path))
            and Path(set_a).suffix != Path(set_b).suffix
        ):
            warnings.warn(
                f"You provided different file types for set A and B. "
                f"Ensure they were generated using the same model ({self.model}).",
                UserWarning,
            )
        logging.info("Resolving input A...")
        emb_a = resolve_input_to_numpy(set_a, "A", self.embedder)
        logging.info("Resolving input B...")
        emb_b = resolve_input_to_numpy(set_b, "B", self.embedder)

        if self.save_embeddings:
            logging.info(f"Saving embeddings to {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.output_dir / "emb_a.npy", emb_a)
            np.save(self.output_dir / "emb_b.npy", emb_b)

        logging.info("Computing mean and covariance...")
        mu_a, sigma_a = get_mu_sigma(emb_a)
        mu_b, sigma_b = get_mu_sigma(emb_b)

        return float(calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b))
