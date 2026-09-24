<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Install the samples with pip

Install the application from the repository using Python 3.11+ and pip. The
repository includes Python source, sample inputs, and the compiled Angular UI.
Only the optional RAG corpus and index are packaged as release assets.

You do not need uv, Node.js, npm, or Codex to run the application. Python packages
must be available through your configured pip package index. Node.js and npm
are needed only when changing and rebuilding the UI.

## Get the repository

Download and extract a source ZIP from the
[repository page](https://github.com/martinbechard/lg-report), or clone it with Git:

```sh
git clone --depth 1 https://github.com/martinbechard/lg-report.git
cd lg-report
```

Git is optional when using the ZIP download. Open a terminal in the repository
root and run all commands there. Keep that directory after installation; it must
be writable for generated reports and caches.

## Install dependencies

**Optional: create a virtual environment.** The following command creates a
folder named `.venv` containing an isolated Python environment. Packages installed
there stay separate from those used by your other Python projects. The application
does not require a virtual environment.

```sh
python -m venv .venv
```

If you created it, activate it with `source .venv/bin/activate` on macOS/Linux, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Activation makes this
terminal's `python` command use that environment. Run `deactivate` to leave it.

If creating or activating a virtual environment does not work on your machine,
you can skip both steps and use an existing Python environment where package
installation is allowed. The commands below then install into that environment.

With your chosen Python environment, run:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[chat]"
python -m pip check
```

**Use the editable installation (`-e`).** The application locates frontend,
pricing, and RAG assets relative to the source tree. A plain wheel installation
does not include this complete layout. Run the commands below from the repository
root, using the same Python environment you installed into. If you chose
the optional virtual environment, activate it again whenever you open a new terminal.

## Start the browser interface

```sh
python -m agent_runtime --sample simple_chat --client angular --demo --prices models.json
```

Open <http://127.0.0.1:8000/>. Uvicorn serves both the prebuilt Angular interface
and the Python API. Stop it with Ctrl-C; use `--port 8001` if needed.
Demo mode uses scripted model responses and requires no provider API key.

For real responses, copy `.env.example` to `.env.local` and set `LG_PROVIDER`,
`LG_MODEL`, and the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
Choose a model available to your account, then run:

```sh
python -m agent_runtime --sample simple_chat --client angular --live --env-file .env.local --prices models.json
```

Live execution incurs provider charges. Shell variables override `.env.local`.
The interface allows selecting the other available samples.

## Console samples and reports

```sh
python -m agent_runtime --list
python -m agent_runtime --sample simple_chat --demo --prices models.json
python scripts/run_samples.py --simulated
```

The individual command writes `reports/simple_chat/report.html`. The batch
writes HTML and Excel files for each local sample, linked by `reports/index.html`.
Rerunning replaces those outputs. No Node.js or Microsoft Excel installation is
needed to generate them.

For a live batch, run `python scripts/run_samples.py --env-file .env.local`.
The quote sample asks for terminal input. Langfuse samples are excluded from
the batch and require their own project credentials; see their sample READMEs.

Before running RAG samples or the complete batch, download the optional archives:

```sh
python scripts/download_rag.py
```

The helper verifies the checkout's pinned checksums and reuses verified pieces.
RAG samples restore the downloaded Chroma index into `.cache/lg-report/rag/`.
The embedding model is not bundled and may download on first use, including
in demo mode. Rebuilding an index also needs a tokenizer download. This is not
an air-gapped installation kit. Live batches refresh the exchange rate online;
pass `--fx-file exchange-rate.json` to use the saved reference instead.

The shell-script sample requires a POSIX host with `sh`. Use macOS, Linux, or
WSL for that sample and the complete batch; native Windows cannot run that
sample. Other samples can be selected individually.

pip resolves the version ranges in `pyproject.toml`; it does not read `uv.lock`.
For repeatable enterprise rollout, retain the approved package versions from
your validated destination environment and use your organization's package index.

## Package RAG assets on the maintainer's machine

Finish ingestion and stop every ingestion or chat process using Chroma before
packaging a RAG cache. From the repository root, run:

```sh
python scripts/package_rag.py
python scripts/package_release.py
```

The first command packages the completed cache as index and dataset archive
pieces under `data/rag/`, with license sidecars and `archives.json`. The second
verifies the catalog-listed pieces, copies them into `dist/`, and writes
`dist/SHA256SUMS`. Neither command builds or bundles application source or UI.
See [RAG setup](data/rag/README.md) for cache requirements and attribution.

To prepare existing release pieces without rebuilding the cache, run
`python scripts/download_rag.py` followed by `python scripts/package_release.py`.

## Publish RAG assets to a GitHub Release

Commit and push the source changes, including the RAG catalog and license
sidecars. Create a draft release against that exact commit. For version 0.1.0,
using the authenticated GitHub CLI:

```sh
gh release create v0.1.0 --target "$(git rev-parse HEAD)" --draft --title "lg-report 0.1.0" --notes "Optional RAG corpus and index. Install the application from the repository; see PIP-INSTALL.md."
```

Upload exactly the RAG pieces named in `data/rag/archives.json`, plus
`dist/SHA256SUMS`. Avoid uploading all of `dist/`: it can contain unrelated builds
or old archive pieces. The checksum file lists the required piece filenames.
Use `gh release upload v0.1.0` followed by those explicit file paths, or select
them in GitHub's draft release page.

Check that every required piece is uploaded before publishing the draft:

```sh
gh release edit v0.1.0 --draft=false
```

For later versions, use the matching tag. Keep archive pieces out of Git;
source and the compiled UI remain in the repository. GitHub may display its
automatically generated source archives alongside the uploaded RAG assets.

After downloading the RAG pieces and checksum file into one directory, verify
them on macOS/Linux with `shasum -a 256 -c SHA256SUMS`. The download helper also
verifies each piece against the repository catalog.
