import pytest
from plm_fid import FrechetProteinDistance
import numpy as np
import torch
from pathlib import Path


@pytest.fixture(scope="session")
def fid():
    return FrechetProteinDistance()


class TestFrechetProteinDistance:
    # Assume ESM2_8M output dim of 320
    @pytest.mark.parametrize(
        "a,b",
        [
            (np.random.rand(10, 320), np.random.rand(10, 320)),
            (torch.rand(10, 320), torch.rand(10, 320)),
            (np.random.rand(10, 320), torch.rand(10, 320)),
        ],
    )
    def test_compute_fid_with_array_tensor_variants(self, fid, a, b):
        result = fid.compute_fid(a, b)
        assert isinstance(result, float)

    def test_compute_fid_with_fasta(self, fasta_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(fasta_file, fasta_file)
        assert isinstance(result, float)

    def test_compute_fid_with_npy(self, npy_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(npy_file, npy_file)
        assert isinstance(result, float)

    def test_compute_fid_with_pt(self, tensor_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(tensor_file, tensor_file)
        assert isinstance(result, float)

    def test_compute_fid_with_fasta_and_npy(self, fasta_file: Path, npy_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(fasta_file, npy_file)
        assert isinstance(result, float)

    def test_compute_fid_with_fasta_and_pt(self, fasta_file: Path, tensor_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(fasta_file, tensor_file)
        assert isinstance(result, float)

    def test_compute_fid_with_npy_and_pt(self, npy_file: Path, tensor_file: Path):
        fid = FrechetProteinDistance(max_length=10, model_name="esm2_8m")
        result = fid.compute_fid(npy_file, tensor_file)
        assert isinstance(result, float)
