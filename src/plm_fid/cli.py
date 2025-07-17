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
    help=(
        "The protein language model to use. Accepts one of the PLM enum values. See --help for available models."
    ),
)
@click.option(
    "--device",
    type=str,
    default=None,
    help=(
        "The device to run the model on, e.g., 'cuda:0' or 'cpu'. "
        "Defaults to 'cuda' if available, otherwise 'cpu'."
    ),
)
@click.option(
    "--max-length",
    type=int,
    default=1000,
    help=(
        "Maximum length for each protein sequence. Longer sequences are truncated "
        "according to the selected truncation style. Some models may require a smaller max length "
        "(e.g., antiberta2_cssp supports up to 254)."
    ),
)
@click.option(
    "--truncation-style",
    type=click.Choice(["end", "center"]),
    default="center",
    help=(
        "How to truncate sequences longer than max length. "
        "'end' truncates from the back, 'center' keeps the central region."
    ),
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Number of sequences to embed per batch.",
)
@click.option(
    "--save-embeddings",
    is_flag=True,
    help=(
        "Whether to save the embeddings used for FID computation to .npy files. "
        "Useful for reuse or debugging."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
    help="Directory to save embeddings if --save-embeddings is enabled.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show progress messages",
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
