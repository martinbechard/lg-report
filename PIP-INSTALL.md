<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Install the prebuilt samples with pip

This bundle includes the compiled Angular interface, Python source, sample
inputs, and bundled RAG archives. The destination needs Python 3.11+ and pip.
It does not need uv, Node.js, npm, Git, or Codex. Python packages must be
available through your configured pip package index.

Download `lg-report-pip.tar.gz` and `SHA256SUMS` from the
[GitHub Releases page](https://github.com/martinbechard/lg-report/releases/latest).
Complete distribution archives are release attachments, not tracked Git files.

## Install on the destination

Extract `lg-report-pip.tar.gz`, open a terminal in its `lg-report` directory,
and keep that directory in place after installation. It must be writable for
generated reports and caches.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[chat]"
python -m pip check
```

**Use the editable installation (`-e`).** The application locates frontend,
pricing, and RAG assets relative to the source tree. A plain wheel installation
does not include this complete layout. Run the commands below from the extracted
directory with the virtual environment activated.

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

RAG samples restore the bundled Chroma index into `.cache/lg-report/rag/`.
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

## Rebuild the distribution on the maintainer's machine

The build machine needs Git, Python, and a Node.js version supported by the
frontend's Angular CLI (Node 22.12+ or Node 24 is suitable for this checkout).
From the repository root:

```sh
python scripts/package_pip.py
```

The script runs `npm ci` and the production Angular build, then creates
`dist/lg-report-pip.tar.gz`. Send that archive to the destination. It includes
current source edits and reference assets, but excludes environment secrets,
virtual environments, dependency caches, and generated conversation reports.
Rebuild it whenever frontend code changes. npm is needed only on this machine.


## Publish a GitHub Release

After rebuilding the pip bundle, build the standard Python distributions and
record SHA-256 checksums. Use the version in `pyproject.toml` for the release tag:

```sh
uv build
python scripts/package_release.py
```

Commit and push the source changes, then create a release against that exact
commit. For version 0.1.0, using the authenticated GitHub CLI:

```sh
gh release create v0.1.0 --target "$(git rev-parse HEAD)" --draft --title "lg-report 0.1.0" --notes "Prebuilt pip bundle and Python distributions. See PIP-INSTALL.md for installation."
gh release upload v0.1.0 dist/lg-report-pip.tar.gz dist/lg_report-0.1.0.tar.gz dist/lg_report-0.1.0-py3-none-any.whl dist/SHA256SUMS
gh release edit v0.1.0 --draft=false
```

Check that every upload succeeded before publishing the draft. For later
versions, update the tag and versioned filenames in these commands. Upload the
complete archives without splitting them. Keep build outputs ignored by Git.
The pip bundle includes the compiled UI and RAG assets; the wheel alone does
not supply the complete source-relative runtime layout.

After downloading all three artifacts and the checksum file into one directory,
verify them on macOS/Linux with `shasum -a 256 -c SHA256SUMS`.
