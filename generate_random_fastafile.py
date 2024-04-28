from __future__ import annotations
import click
import random
from typing import List
from pathlib import Path
import uuid

from minsh.utils import save_fasta_file

@click.command()
@click.option(
    "--line-length",
    "-ll",
    default=100,
    help="The length of the lines in the fasta file.",
)
@click.option(
    "--length",
    "-l",
    default=[100, 1_000, 10_000, 100_000],
    multiple=True,
    help="The length of the random sequence.",
)
@click.option(
    "--copies",
    "-c",
    default=5,
    help="The number of copies of each length to generate.",
)
@click.option(
    "--output-prefix", "-o-p", default=None, help="The output file name prefix."
)
@click.option(
    "--alphabet",
    "-a",
    default="ACGT",
    help="The alphabet to use for the random sequence.",
)
@click.option(
    "--description",
    "-d",
    default="Autogenerated",
    help="The description of the fasta file.",
)
@click.option(
    "--clobber", "-clb", is_flag=True, help="Overwrite the output file if it exists."
)
@click.option(
    "--output-directory",
    "-id",
    default="data",
    help="The directory to save the files in.",
)
def main(
    line_length: int,
    length: List[int],
    copies: int,
    output_prefix: str,
    alphabet: str,
    description: str,
    clobber: bool,
    output_directory: str,
) -> None:
    assert len(length) > 0
    assert line_length > 0
    assert len(alphabet) == len(set(alphabet)) and len(alphabet) > 0

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Pick this length because it'll be more readable in the FS (arbitrary)
    uuid_prefix_len = 5
    output_prefix = f"{uuid.uuid4()}"[:uuid_prefix_len] if output_prefix is None else output_prefix
    assert len(output_prefix) > 0
    for length in length:
        for i in range(copies):
            sequence = "".join(random.choice(alphabet) for _ in range(length))
            output_file = output_directory / f"{output_prefix}_l{length}_v{i}.fa"
            if not clobber and output_file.exists():
                # Uncomment to take a more "silent failure" approach
                # print(f"Skipping {output_file} as it already exists.")
                # continue
                raise FileExistsError(f"{output_file} already exists.")
            save_fasta_file(output_file, description, sequence, line_length)
            click.echo(f"Saved {output_file}")



if __name__ == "__main__":
    main()
