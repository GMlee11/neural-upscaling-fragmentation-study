# Evidence guide

- Original RK3576 paired-control table: results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json; src/report/paper3_rk3576_fragmentation_pairs_v1.py; configuration and dated protocol under configs/ and docs/.
- Second-graph RK3576/RK3566 table: results/manifests/paper3_shared_v2_fp16_*capture_v22.json and *capture_v24.json; run_paper3_shared_v2_fp16_v22.py and run_paper3_shared_v2_fp16_rk3566_v24.py, offline `analyze` commands only.
- Width-24 RK3566 cell: results/manifests/paper3_width24_int8_rk3566_public_v32.json; src/report/paper3_width24_rk3566_v32.py.
- Organization-tax and 15-mask description: results/tables/codesign_matched_masks_rk3576_v1_public.json; src/report/paper3_npu_fragmentation_summary_v1.py.
- Dispatch outcomes and per-request decisions: results/manifests/paper3_dispatch_public_v1.json; src/report/paper3_dispatch_publication_v1.py; artifact/paper3_samos/DISPATCH_REPLAY.md. This embeds protocol, calibration identities and all 5,120 projected records.
- Separate zero-target-fitting transfer: results/manifests/paper3_transfer_public_v1.json; src/report/paper3_transfer_publication_v1.py; artifact/paper3_samos/TRANSFER_REPLAY.md. This adds all 5,120 transfer records and the frozen donor, with no target fit.
- Complete-service/reconstruction-placement confirmation: results/paper3_strengthening_v2/public_v1/ and artifact/paper3_samos/STRENGTHENING_REPLAY.md. Four exact journals retain 1,664 evaluations, 192 qualifications and four references, with source/method/masks/profiles, recorded controls, original calibration and hash-bound private audits. Compact replay regenerates all planned practical gates and prediction diagnostics; it does not reopen omitted raw tensors.
- Confirmation service and geometry tables: src/report/paper3_strengthening_publication.py regenerates all three paper3_strengthening_*_v1 table/JSON files under docs/paper/generated/ from the exact saved-record audit. Table fragments are allowed; no manuscript is distributed.
- Earlier RK3566 packing-development context: artifact/paper3_samos/PACKING_CONTEXT.md and replay_packing_context.py validate all 176 observations and eleven fixed-organization means from the retained original source/journals. This is not the later RK3576 confirmation, a stable packing ranking, or compact pixel revalidation.

The reproduction script lists exact commands and compares generated JSON, Markdown
and table fragments byte-for-byte. Tests cover analysis and decision replay, not
manuscript strings. Original graphs, compiled RKNN files, reference tensors and
raw output arrays remain outside the compact release; E002 journals are included.
Omitted payload hashes and
qualification summaries are provenance, not substitutes for rerunning numerical
qualification. No FPGA normalization or result from a companion paper enters
the analysis. Do not rerun deployment entry points as part of artifact review.
