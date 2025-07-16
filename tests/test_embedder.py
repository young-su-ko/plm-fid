import pytest
from plm_fid.embedding import ProteinEmbedder
from pathlib import Path
from plm_fid.models import PLM
import torch


class TestProteinEmbedder:
    @pytest.mark.parametrize(
        "model",
        [
            PLM.ESM2_650M,
            PLM.ESM2_150M,
            PLM.PROTBERT,
            PLM.ESMPLUSPLUS_SMALL,
            PLM.ANTIBERTA2_CSSP,
        ],
    )
    def test_load_model_and_tokenizer(self, model: PLM):
        embedder = ProteinEmbedder(model=model)
        model, tokenizer = embedder._load_model_and_tokenizer(model)
        assert model is not None
        assert tokenizer is not None

    def test_end_truncation(self):
        embedder = ProteinEmbedder(truncation_style="end")
        test_sequence = "MALWMRLLPLLALLALWGPDPAAA" * 1000
        assert embedder._truncate_sequence(test_sequence) == test_sequence[:1000]

    def test_center_truncation(self):
        embedder = ProteinEmbedder(truncation_style="center", max_length=3)
        test_sequence = "PAAAP"
        assert embedder._truncate_sequence(test_sequence) == "AAA"

    def test_antiberta2_cssp_max_length(self):
        embedder = ProteinEmbedder(model=PLM.ANTIBERTA2_CSSP, max_length=1000)
        test_sequence = "MALWMRLLPLLALLALWGPDPAAA" * 1000
        assert len(embedder._truncate_sequence(test_sequence)) == 254

    def test_embed_fasta(self, fasta_file: Path):
        embedder = ProteinEmbedder()
        embeddings = embedder._embed_fasta(fasta_file)
        # using default ESM2_650M, the output shape is (2, 1280)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 1280
