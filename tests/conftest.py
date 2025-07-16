import pytest
import tempfile
from pathlib import Path
import numpy as np
import torch


def temporary_fasta_file(content: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_file:
        temp_file.write(content.encode())
        return Path(temp_file.name)


def temporary_npy_file(content: np.ndarray):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as temp_file:
        np.save(temp_file.name, content)
        return Path(temp_file.name)


def temporary_tensor_file(content: torch.Tensor):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        torch.save(content, temp_file.name)
        return Path(temp_file.name)


@pytest.fixture
def fasta_file():
    content = """>1
MALWMRLLPLLALLALPDPAAA
>2
MALWMRLLPLLALLALWPDPAAA
>3
MALWMLALWGPDPAAA
>4
MALWMRLLPLDPAAA
>5
MAPLLALLALWGPDPAAA
>6
MALWMRLLPLL
"""
    return temporary_fasta_file(content)


@pytest.fixture
def npy_file():
    content = np.random.rand(10, 320)
    return temporary_npy_file(content)


@pytest.fixture
def tensor_file():
    content = torch.rand(10, 320)
    return temporary_tensor_file(content)
