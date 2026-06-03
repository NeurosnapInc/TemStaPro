"""CLI tests for TemStaPro focused on stable, user-visible behavior."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMSTAPRO_BIN = REPO_ROOT / "temstapro"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the TemStaPro CLI and capture combined stdout and stderr."""
    return subprocess.run(
        [sys.executable, str(TEMSTAPRO_BIN), *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _table_lines(output: str) -> list[str]:
    """Return TSV table lines from combined CLI output."""
    return [line for line in output.splitlines() if "\t" in line]


def test_cli_requires_fasta_argument() -> None:
    """The CLI should fail with a clear message when FASTA input is omitted."""
    result = _run_cli("-d", "./ProtTrans/")

    assert result.returncode != 0
    assert "a FASTA file is required." in result.stdout


def test_cli_requires_prottrans_directory() -> None:
    """The CLI should fail with a clear message when the ProtTrans path is omitted."""
    result = _run_cli("-f", "tests/data/multiple_short_sequences.fasta")

    assert result.returncode != 0
    assert "a path to the ProtTrans model location is required." in result.stdout


def test_cli_replaced_symbols_run_succeeds(tmp_path: Path) -> None:
    """A small sequence with replaced symbols should complete and write both outputs."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mean_output = tmp_path / "mean.tsv"
    per_res_output = tmp_path / "per_res.tsv"

    result = _run_cli(
        "-f", "tests/data/replaced_symbol_sequence.fasta",
        "-e", str(cache_dir),
        "-d", "./ProtTrans/",
        "--mean-output", str(mean_output),
        "--per-res-output", str(per_res_output),
    )

    assert result.returncode == 0, result.stdout
    assert "beginning to load the model" in result.stdout
    assert "finished making inferences" in result.stdout
    assert mean_output.exists()
    assert per_res_output.exists()

    mean_rows = _read_tsv(mean_output)
    per_res_rows = _read_tsv(per_res_output)
    assert len(mean_rows) == 1
    assert len(per_res_rows) == 1
    assert mean_rows[0]["protein_id"] == "artificial_sequence"
    assert per_res_rows[0]["protein_id"] == "artificial_sequence"
    assert mean_rows[0]["sequence"]
    assert per_res_rows[0]["sequence"]


def test_cli_missing_embeddings_dir_warns_but_still_runs(tmp_path: Path) -> None:
    """A missing cache directory should warn and still produce predictions."""
    missing_cache_dir = tmp_path / "missing-cache"
    mean_output = tmp_path / "mean.tsv"

    result = _run_cli(
        "-f", "tests/data/multiple_short_sequences.fasta",
        "-e", str(missing_cache_dir),
        "-d", "./ProtTrans/",
        "--mean-output", str(mean_output),
    )

    assert result.returncode == 0, result.stdout
    assert "does not exist" in result.stdout
    assert "will not be saved" in result.stdout
    assert mean_output.exists()

    mean_rows = _read_tsv(mean_output)
    assert len(mean_rows) == 3


def test_cli_reuses_cached_embeddings_on_second_run(tmp_path: Path) -> None:
    """A second run should reuse cached embeddings instead of regenerating them."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    first_mean_output = tmp_path / "first_mean.tsv"
    second_mean_output = tmp_path / "second_mean.tsv"

    first = _run_cli(
        "-f", "tests/data/multiple_short_sequences.fasta",
        "-e", str(cache_dir),
        "-d", "./ProtTrans/",
        "--mean-output", str(first_mean_output),
    )
    assert first.returncode == 0, first.stdout
    assert "beginning to generate embeddings" in first.stdout

    cached_embeddings = list(cache_dir.glob("*.pt"))
    assert cached_embeddings

    second = _run_cli(
        "-f", "tests/data/multiple_short_sequences.fasta",
        "-e", str(cache_dir),
        "-d", "./ProtTrans/",
        "--mean-output", str(second_mean_output),
    )

    assert second.returncode == 0, second.stdout
    assert "beginning to generate embeddings" not in second.stdout
    assert "beginning to make inferences" in second.stdout
    assert first_mean_output.read_text(encoding="utf-8") == second_mean_output.read_text(encoding="utf-8")


def test_cli_more_thresholds_adds_extended_columns(tmp_path: Path) -> None:
    """The extended thresholds mode should print the extra threshold columns."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result = _run_cli(
        "-f", "tests/data/multiple_short_sequences.fasta",
        "-e", str(cache_dir),
        "-d", "./ProtTrans/",
        "--more-thresholds",
    )

    assert result.returncode == 0, result.stdout

    lines = _table_lines(result.stdout)
    assert lines
    header = lines[0].split("\t")
    assert "t70_binary" in header
    assert "t75_binary" in header
    assert "t80_binary" in header
    assert "thermophilicity" in header
    assert len(lines[1:]) == 3


def test_cli_per_residue_plots_are_written(tmp_path: Path) -> None:
    """Per-residue output mode should generate one SVG plot per short input sequence."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mean_output = tmp_path / "mean.tsv"
    per_res_output = tmp_path / "per_res.tsv"
    plot_dir = tmp_path / "plots"

    result = _run_cli(
        "-f", "tests/data/multiple_short_sequences.fasta",
        "-e", str(cache_dir),
        "-d", "./ProtTrans/",
        "--mean-output", str(mean_output),
        "--per-res-output", str(per_res_output),
        "-p", str(plot_dir),
    )

    assert result.returncode == 0, result.stdout
    assert mean_output.exists()
    assert per_res_output.exists()
    assert plot_dir.exists()
    assert len(list(plot_dir.glob("*.svg"))) == 3
