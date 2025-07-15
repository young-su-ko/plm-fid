import click
from pathlib import Path
import logging

from .api import FrechetProteinDistance
from .models import PLM


@click.command(
    help=(
        "Compute Fréchet distance between two protein sets (.fasta or .npy).\n\n"
        "Examples:\n\n"
        "  plm-fid set1.fasta set2.fasta\n\n"
        "  plm-fid set1.npy set2.npy --save-embeddings\n\n"
        "  plm-fid set1.fasta set2.npy --model antiberta2_cssp\n"
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("set_a", type=click.Path(exists=True, path_type=Path))
@click.argument("set_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    type=click.Choice([p for p in PLM], case_sensitive=False),
    default=PLM.ESM2_650M,
    help="Protein language model to use for embedding, if using .fasta files",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to run the pLM on, e.g., 'cuda:0' or 'cpu'.",
)
@click.option(
    "--max-length",
    type=int,
    default=1000,
    help="Maximum sequence length for the pLM. Note: Some models have specific length constraints (e.g., antiberta2-cssp requires max_length ≤ 254).",
)
@click.option(
    "--truncation-style",
    type=click.Choice(["end", "center"]),
    default="end",
    help="Truncation style for long sequences. End truncates the sequence from the end, center truncates from the center.",
)
@click.option(
    "--batch-size", type=int, default=1, help="Batch size for protein embedding"
)
@click.option(
    "--save-embeddings",
    is_flag=True,
    help="Save computed embeddings as an .npy file. Useful if you want to reuse embeddings in future runs.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
    help="Directory to save output files. If --save-embeddings is used, the embeddings will be saved in this directory.",
)
@click.option("--verbose", is_flag=True, help="Show progress messages")
def main(
    set_a,
    set_b,
    model,
    device,
    max_length,
    truncation_style,
    batch_size,
    save_embeddings,
    output_dir,
    verbose,
):
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    fid = FrechetProteinDistance(
        model=model,
        device=device,
        max_length=max_length,
        truncation_style=truncation_style,
        batch_size=batch_size,
        save_embeddings=save_embeddings,
        output_dir=output_dir,
    )

    distance = fid.compute_fid(set_a, set_b)
    click.echo(f"FID: {distance}")


if __name__ == "__main__":
    main()
