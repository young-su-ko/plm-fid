import numpy as np
import torch
from pathlib import Path
from .embedding import ProteinEmbedder


def check_embedding_shape(emb: np.ndarray, name: str) -> None:
    if len(emb.shape) != 2:
        raise ValueError(
            f"Embedding {name} must be 2D with shape [batch_size, embedding_dim], got shape {emb.shape}"
        )


def load_embeddings_from_path(path: Path):
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".pt":
        return torch.load(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def resolve_input_to_numpy(
    input_set: str | Path | torch.Tensor | np.ndarray,
    set_name: str,
    embedder: ProteinEmbedder,
) -> np.ndarray:
    """
    Convert an input object to a NumPy array of embeddings.

    This function handles multiple input types:
    - A `.fasta` file will be embedded using the provided `embedder`
    - A `.npy` or `.pt` file will be loaded as precomputed embeddings
    - A PyTorch tensor will be converted to a NumPy array
    - A NumPy array will be returned as-is

    Parameters
    ----------
    input_set : str or Path or np.ndarray or torch.Tensor
        Input source for a protein set. Must be one of the following:
        - A path to a `.fasta`, `.npy`, or `.pt` file
        - A NumPy array of shape (N, D)
        - A PyTorch tensor of shape (N, D)

    set_name : str
        Identifier for the set (e.g., "A" or "B"). Used in error messages.

    embedder : ProteinEmbedder
        An initialized embedder used to generate embeddings from FASTA files.
        Is unused if the input is a NumPy array or PyTorch tensor.

    Returns
    -------
    embedding : np.ndarray
        A 2D NumPy array of shape (N, D) representing the set of protein embeddings.

    Raises
    ------
    ValueError
        If the file suffix is unsupported, or the resulting embedding is not 2D.

    TypeError
        If the input type is not supported (e.g., dict, list, etc.).

    Examples
    --------
    >>> resolve_input_to_numpy("proteins.fasta", "A", embedder)
    array([[...], [...]])

    >>> resolve_input_to_numpy("precomputed.npy", "B", embedder)
    array([[...], [...]])

    >>> resolve_input_to_numpy(np.random.randn(10, 128), "A", embedder)
    array([[...], [...]])
    """
    if isinstance(input_set, (np.ndarray, torch.Tensor)):
        embedding = input_set
    elif isinstance(input_set, (str, Path)):
        path = Path(input_set)
        if path.suffix in [".npy", ".pt"]:
            embedding = load_embeddings_from_path(path)
        elif path.suffix == ".fasta":
            embedding = embedder._embed_fasta(path)
        else:
            raise ValueError(f"Unsupported file type for set {set_name}: {path.suffix}")
    else:
        raise TypeError(f"Unsupported input type for set {set_name}: {type(input_set)}")

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()

    check_embedding_shape(embedding, set_name)
    return embedding


def get_mu_sigma(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and covariance of a set of embeddings using numpy.

    Parameters
    ----------
    embeddings : np.ndarray
        A 2D array of shape (N, D) representing the set of embeddings.

    Returns
    -------
    mu : np.ndarray
        The mean vector of shape (D,).

    sigma : np.ndarray
        The covariance matrix of shape (D, D).
    """
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma
