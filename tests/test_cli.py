import pytest
from click.testing import CliRunner
from plm_fid.cli import main
from pathlib import Path
import numpy as np


@pytest.fixture
def runner():
    return CliRunner()


class TestCli:
    def test_cli_with_fasta_inputs(self, runner, fasta_file: Path):
        result = runner.invoke(
            main,
            [
                str(fasta_file),
                str(fasta_file),
                "--verbose",
                "--model-name",
                "esm2_8m",
                "--max-length",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "FID:" in result.output

    def test_cli_with_npy_inputs(self, runner, tmp_path: Path):
        file_a = tmp_path / "a.npy"
        file_b = tmp_path / "b.npy"
        np.save(file_a, np.random.rand(10, 320))
        np.save(file_b, np.random.rand(10, 320))

        result = runner.invoke(main, [str(file_a), str(file_b), "--verbose"])
        assert result.exit_code == 0
        assert "FID:" in result.output

    def test_cli_with_mismatched_filetypes_warns(
        self, runner, fasta_file: Path, tmp_path: Path
    ):
        file_b = tmp_path / "b.npy"
        np.save(file_b, np.random.rand(10, 320))

        result = runner.invoke(
            main,
            [
                str(fasta_file),
                str(file_b),
                "--verbose",
                "--model-name",
                "esm2_8m",
                "--max-length",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "FID:" in result.output
