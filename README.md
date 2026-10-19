# Fragmentation Tax in Sparse Neural Upscaling Across Two Rockchip NPUs - reviewer artifact

Grant Lee, University of Central Florida, gr811059@ucf.edu
ORCID: https://orcid.org/0009-0009-9377-8412
Prepared for Journal of Signal Processing Systems. Version review-v1, 2026-09-25.

This is an artifact-only repository: measurements, evidence, analysis code and
tests, with no manuscript, preprint or submission correspondence. Publication
preparation is not a claim that the paper has been accepted. Legacy `paper3_samos`
path names are stable experiment/replay identifiers, not the current venue.

## Review without hardware

Use Python 3.12 and Windows PowerShell 5.1 (the replay entry point is PowerShell).
Do not run deployment scripts on a board for this review. No hardware, FPGA
toolchain, neural-network compiler, trained weights or LaTeX is needed.

```powershell
py -3.12 -m venv ..\artifact-review-env
..\artifact-review-env\Scripts\python.exe -m pip install -r artifact/paper3_samos/requirements.txt
..\artifact-review-env\Scripts\python.exe -B verify_release.py
powershell -NoProfile -ExecutionPolicy Bypass -File artifact/paper3_samos/reproduce.ps1 -Python (Resolve-Path ..\artifact-review-env\Scripts\python.exe)
```

The environment is intentionally outside this exact-inventory release. Replay
uses a disposable external copy and preserves the released evidence. On another
OS, install PowerShell or execute the documented Python analysis commands; the
full convenience script is validated on Windows. Standard-library hash verification
does not require installing any packages. `.gitattributes` preserves byte identities
through Git; do not enable automatic line-ending conversion for this snapshot.

## Evidence and limitations

The earlier per-cell-calibrated dispatch projection includes 5,120 timing/decision records. A separate 5,120-request transfer projection freezes one donor's coefficients without target fitting; see artifact/paper3_samos/TRANSFER_REPLAY.md. E002 adds complete-service/reconstruction-placement confirmation: all 1,664 evaluation requests, 192 qualifications and four references in four exact journals, frozen source/method/masks/profile/chooser records, and original calibration journals. Compact replay recomputes timing/control checks, choices, all planned practical gates, prediction diagnostics and service/geometry tables; see artifact/paper3_samos/STRENGTHENING_REPLAY.md. These are distinct studies, not pooled. Full output tensors, reference arrays, compiled models and vendor runtimes are excluded. Hash-bound private numerical reports are retained; compact replay does not recompute omitted pixels. Individual regressions, session structure and applicability limitations remain visible.

See EVIDENCE_GUIDE.md for table-to-evidence mapping. Generated `.tex` files are
table fragments, not manuscript sources. The archive preserves unsuccessful and
null outcomes. Do not combine historical diagnostic and confirmatory campaigns.

Only original code is Apache-2.0; original data/documentation is CC BY 4.0.
See LICENSE and THIRD_PARTY_NOTICES.md. `release_manifest.json` hashes every
distributed file other than itself. FILES.txt includes the manifest. Extra files
other than root `.git` metadata cause verification failure deliberately.
