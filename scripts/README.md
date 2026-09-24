<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; AI assistance: Northstar. -->
# Scripts

These scripts run teaching samples, maintain reference data, and prepare distribution files. Run all commands from the repository root with Python 3.11 or later.

Install project dependencies before using scripts that import the application:

```sh
uv sync --locked
```

Examples use `uv run python`. With an activated environment containing the project, use `python` instead. See the [project README](../README.md) for setup and [pip installation instructions](../PIP-INSTALL.md) for a pip-only environment.

## Using pip without uv

You can run every script without installing uv. From the repository root, optionally create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. If your system uses `python3`, use that command to create the environment. Alternatively, use an existing Python 3.11+ environment where installation is allowed.

Install the project into your chosen environment:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[chat]"
python -m pip check
```

The editable installation (`-e`) keeps repository-relative sample and data files available. The `chat` extra also installs the browser server dependencies; use `-e .` if you only need the scripts. Pip resolves dependencies from `pyproject.toml`; it does not reproduce the exact versions in `uv.lock`.

Replace `uv run python` with `python` in every script command below. Keep all script arguments unchanged. For example:

```sh
python scripts/download_rag.py
python scripts/run_samples.py --simulated --out reports/simulated
python scripts/calibrate_models.py --help
```

Use the same Python environment for installation and execution. Activate it again when opening another terminal. Git, Node.js, and npm are still required when running `package_pip.py`.

## Script directory

```text
scripts/
├── run_samples.py             # Run local samples and export reports
├── download_rag.py            # Download and verify RAG archive pieces
├── calibrate_models.py        # Verify configured model metadata
├── update_exchange_rate.py    # Refresh the saved USD-to-EUR rate
├── render_design_diagrams.py  # Regenerate SVG figures in the design page
├── package_rag.py             # Archive a completed RAG cache
├── package_pip.py             # Build the pip installation bundle
└── package_release.py         # Collect RAG pieces and write release checksums
```

## Run samples and export reports

[run_samples.py](run_samples.py) runs every sample registered with local tracing. It excludes Langfuse samples and produces HTML and Excel reports.

```sh
# Real provider calls; reads .env.local by default.
uv run python scripts/run_samples.py

# Scripted models, with results in a separate directory.
uv run python scripts/run_samples.py --simulated --out reports/simulated

# Explicit settings and saved references.
uv run python scripts/run_samples.py --env-file .env.local --prices models.json --fx-file exchange-rate.json
```

Live runs require `LG_PROVIDER=openai` and `OPENAI_API_KEY`, or `LG_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. Shell values override the environment file. Live calls can incur provider charges; the script does not fall back to simulation.

| Option | Meaning |
| --- | --- |
| `--out PATH` | Output directory; defaults to `reports`. Reruns replace generated results in each sample directory. |
| `--simulated` | Use scripted models and skip the exchange-rate refresh. |
| `--env-file PATH` | Credentials and model settings; defaults to the repository's `.env.local`. |
| `--prices PATH` | Saved model/pricing catalog; defaults to the repository's `models.json`. No prices are fetched. |
| `--fx-file PATH` | Saved USD-to-EUR reference; skips the exchange-rate refresh. `LG_FX_FILE` also selects a saved reference. |

Download the RAG archives before a complete batch, using the next script. Simulated RAG samples still perform real local retrieval. Their first use may download an embedding model.

The live `quote_request` sample can ask questions in the terminal. The batch automatically approves the `file_approval` sample's edit within its report directory.

Open `index.html` in the output directory to inspect results. Each sample directory contains reports, accounting evidence, and `run.log`. Sample failures do not stop later samples; the batch returns exit code 1 if any sample fails.

## Download RAG archives

[download_rag.py](download_rag.py) downloads archive pieces listed in the [RAG checksum catalog](../data/rag/archives.json) into the catalog's directory. It verifies sizes and SHA-256 checksums, and reuses valid existing pieces.

```sh
uv run python scripts/download_rag.py

# Select the release matching an older checkout.
uv run python scripts/download_rag.py --tag v0.1.0
```

The default source is the latest GitHub Release. This script uses only Python's standard library, so `python scripts/download_rag.py` also works before dependency installation.

Downloads can be retried after interruption. The script does not extract archives or download embedding models. RAG samples restore the index on first use; see [RAG setup](../data/rag/README.md).

## Calibrate model metadata

[calibrate_models.py](calibrate_models.py) checks each real model in a catalog and saves access and context-capacity evidence in its `calibrations` section.

```sh
uv run python scripts/calibrate_models.py
uv run python scripts/calibrate_models.py --config models.json
```

`--config PATH` selects the file to update; the default is `models.json`. Credentials come from the shell or `.env`, with shell values taking precedence. This differs from the batch runner's default `.env.local`.

Provide `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` for the providers present in the catalog. The script queries provider metadata and, for OpenAI, published model documentation. It makes no generation requests and does not change prices.

Failed checks replace previous calibration evidence with an explicit unavailable capacity. Results are saved even when some checks fail; exit code 1 signals a failure. Metadata checks do not prove inference permission or maximum-prompt acceptance.

## Refresh the exchange rate

[update_exchange_rate.py](update_exchange_rate.py) fetches a USD-to-EUR reference from Frankfurter using ECB data and saves its provenance.

```sh
uv run python scripts/update_exchange_rate.py
uv run python scripts/update_exchange_rate.py --out reports/reference/exchange-rate.json
```

`--out PATH` defaults to the repository's `exchange-rate.json`. A failed request or invalid response preserves the previous file and returns exit code 1. No model credentials are required.

Live batches invoke this script once unless a saved rate is selected. Individual samples and report rendering consume saved rates. Missing valid rates leave EUR costs unknown.

## Render design diagrams

[render_design_diagrams.py](render_design_diagrams.py) replaces the 20 embedded diagram figures in the [technical design page](../docs/index.html) with SVG generated from its Python definitions.

```sh
uv run python scripts/render_design_diagrams.py
```

Run this after editing diagram definitions. It writes the design page in place and requires no browser or external rendering service. It accepts no command-line options.

## Package a RAG cache

[package_rag.py](package_rag.py) compresses a completed local RAG cache into separate index and dataset archives. Before running it, finish ingestion and stop every ingestion or chat process using Chroma.

```sh
uv run python scripts/package_rag.py
```

The script reads `.cache/lg-report/rag/`, including its completion manifest, Chroma database, configuration, and WikiText dataset. It writes archive pieces of at most 48 MiB, license sidecars, and an updated checksum catalog under `data/rag/`.

Packaging replaces matching pieces and rewrites `archives.json`. The complete archive set is not replaced atomically, so do not run consumers during packaging. Old, unreferenced pieces are not removed automatically. See [RAG packaging guidance](../data/rag/README.md) for attribution and cleanup.

This script accepts no command-line options and starts packaging immediately when invoked, including with `--help`.

## Build a pip installation bundle

[package_pip.py](package_pip.py) builds the Angular frontend and packages current source files with the compiled UI.

```sh
uv run python scripts/package_pip.py
```

The build machine needs Git, Node.js compatible with the frontend's Angular CLI, and npm. The script runs `npm ci` and `npm run build` in the frontend directory, then writes `dist/lg-report-pip.tar.gz`.

The bundle includes current uncommitted source edits. It excludes real environment files, dependency caches, generated conversation reports, and RAG archive pieces. Recipients download RAG pieces separately and use an editable pip installation; see [PIP-INSTALL.md](../PIP-INSTALL.md).

A failed install or build preserves the previous bundle. This script accepts no command-line options.

## Prepare release checksums

[package_release.py](package_release.py) verifies RAG pieces against the catalog, copies them into `dist/`, and writes `dist/SHA256SUMS`.

It requires the pip bundle, Python source distribution, wheel, and every catalog-listed RAG piece. Obtain RAG pieces with `download_rag.py` or create them with `package_rag.py`.

```sh
uv run python scripts/package_pip.py
uv build
uv run python scripts/package_release.py
```

With pip only, install the Python build frontend and use `python -m build` in place of `uv build`:

```sh
python -m pip install build
python scripts/package_pip.py
python -m build
python scripts/package_release.py
```

Both build commands produce the source distribution and wheel in `dist/` using the project's configured build backend.

The checksum file covers the pip bundle, versioned source archive, versioned wheel, and RAG pieces. The version comes from `pyproject.toml`. Missing artifacts or invalid RAG pieces stop the script.

This script accepts no command-line options. It prepares local release files; publishing is a separate procedure in [PIP-INSTALL.md](../PIP-INSTALL.md).

## Command help

The four scripts with argument parsers support `--help`:

```sh
uv run python scripts/run_samples.py --help
uv run python scripts/download_rag.py --help
uv run python scripts/calibrate_models.py --help
uv run python scripts/update_exchange_rate.py --help
```

The diagram and packaging scripts have no help mode; invoking them performs their normal work.
