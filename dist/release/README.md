<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Rebuilt distributions

The complete archives exceed GitHub's per-file limit. These numbered parts
preserve their exact bytes; use the pip bundle for a source installation with
the prebuilt browser. Licensing is unchanged.

From this directory on macOS/Linux, verify and reconstruct the archives:

```sh
shasum -a 256 -c PARTS.sha256
cat lg-report-pip.tar.gz.part* > ../lg-report-pip.tar.gz
cat lg_report-0.1.0.tar.gz.part* > ../lg_report-0.1.0.tar.gz
cat lg_report-0.1.0-py3-none-any.whl.part* > ../lg_report-0.1.0-py3-none-any.whl
shasum -a 256 -c ARCHIVES.sha256
```

Send `dist/lg-report-pip.tar.gz` with the instructions in `PIP-INSTALL.md`.
The plain wheel does not supply the source-relative runtime assets.

To regenerate from the repository root with compatible Node.js and uv:

```sh
python3 scripts/package_pip.py
uv build
python3 scripts/package_release.py
```
