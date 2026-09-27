# Paper 3 Literature and Novelty Matrix

Last searched: 2026-08-12

## Search Boundary

Paper 3 is a bounded RK3576/RK3566 RKNN fragmentation case study. It does **not** claim
that heterogeneous super-resolution, tile selection, sparse CNN execution,
deadline scheduling, or FPGA acceleration are individually new. The candidate
contribution is the controlled physical comparison of how one upscaling
request behaves through:

1. dense static RKNN graphs;
2. repeated independent sparse graph launches;
3. row-span graph compaction.

The live analysis is self-contained: selected work comes from Paper 3 masks,
while dispatched work and tensor traffic come from the RKNN shapes actually
submitted. No Paper 2 counter or external accelerator denominator is used.

The primary comparison uses one checkpoint identity, resolution, mask geometry,
tile geometry, nominal precision, and frame deadline. A second preregistered
cell changes graph topology and precision together while retaining the request,
runtime family, and analysis contract; the graph cells are then replicated on
RK3566. This tests bounded transfer, not raw INT8/FP16 or cross-device ranking.
Backend quantization and pixel values are intentionally not claimed identical.

## Closest Prior Systems

| Work | Venue/year | What it contributes | Execution granularity | Physical platform | What overlaps | What remains different in Paper 3 |
|---|---:|---|---|---|---|---|
| MobiSR | MobiCom 2019 | Difficulty-aware scheduling across heterogeneous mobile processors | Image patches and model/engine pairs | Mobile CPU/GPU/DSP class processors | Heterogeneous mobile SR and patch scheduling | No equal-count/equal-row static-RKNN fragmentation control |
| SRNPU | JETCAS 2020 | Dedicated mobile SR processor with tile-based selective SR | Tile and network size | Custom SR processor | Selective tile execution and specialized SR hardware | No matched commodity-RKNN launch-versus-padding characterization |
| SMSR | CVPR 2021 | Learned spatial and channel masks for efficient SR | Spatial regions and channels | GPU-oriented algorithm evaluation | Spatially sparse SR | Algorithmic sparsity, not the measured deployment-interface boundary studied here |
| NAWQ-SR | IEEE TMC 2024 | Hybrid-precision NPU execution and runtime neural image codec | Layer/sample precision | Mobile NPU | NPU-specific SR optimization | Focuses precision and quality rather than fragmented worklists, launch overhead, or FPGA persistence |
| Selective SR | AAAI 2024 | Reconstructs selected image tiles with deep transformer features | Multi-scale tiles | GPU algorithm evaluation | Selected-tile reconstruction | Does not measure compact-versus-fragmented static RKNN dispatch at fixed selected work |
| TileSR | INFOCOM 2024 | Parallel offload at tile granularity | Tiles | Device/edge system | Tile-level organization and communication | Offloading differs from local RKNN launch, padding, and tensor-traffic effects |
| FYE-SR | MobiCom 2024 | Perceptual-quality-aware split across heterogeneous mobile processors | Model partition/intermediate features | Mobile heterogeneous processors | Mobile SR co-execution | No physical FPGA counterpart or exact same-mask organization comparison |
| SkipVSR | ACM MM 2024 | Adaptive patch routing using inter-frame masks | Video patches | GPU algorithm evaluation | Temporal patch routing | Paper 3 freezes the selector as an adverse result and studies arbitrary external masks |
| ESSR | IEEE TCAS-I 2024 | Edge-selective network and high-throughput custom SR accelerator | Patches/subnets | 28-nm ASIC implementation report | Edge-driven dynamic SR hardware | No commodity-RKNN fragmentation control; journal metadata verified at DOI 10.1109/TCSI.2024.3349581 |
| SkipSR | Disclosed preprint 2025 | Bypasses low-detail tokens in diffusion video SR | Tokens/patches | GPU | Selective refinement and compute skipping | Different model class; no NPU/FPGA matched-mask hardware boundary |
| EOS | MobiCom 2025 | Energy-optimized mobile SR for live 360-degree video | Runtime system/model choices | Mobile heterogeneous platform | Mobile energy/runtime co-design | Paper 3 currently makes no physical-energy claim and focuses sparse organization |
| DeSpa | JSA 2026 | Dense/sparse heterogeneous accelerator cores at tile level | Tiles | Accelerator model/evaluation | Dense and sparse engine coordination | General sparse DNN engine, not a paired RKNN request-organization study |
| SCNN / NullHop / Eyeriss v2 | ISCA 2017 / TNNLS 2019 / JETCAS 2019 | General sparse-CNN dataflows | Values, activations, channels | Custom accelerators | Sparse execution and data reuse | Value/activation sparsity differs from externally selected output regions and static-shape fragmentation |
| PREMA / ART | HPCA 2020 / GLSVLSI 2025 | Preemptible NPU task scheduling / deadline-customized accelerators | Tasks/jobs | NPU or custom accelerator | Deadlines, admission, and scheduling | Paper 3 measures within-request graph fragmentation and claims no new scheduler or accelerator |
| Hardware-aware DSA graph scheduling | OSDI 2023 | Coarse hardware-aware subgraphs form fewer/larger kernels and reduce off-core movement | Graph subgraphs/kernels | DSA and GPU study | Kernel formation, launch and traffic amortization | Prior interface remedy; Paper 3 contributes only the measured RKNN case study |
| NPUMoE | arXiv preprint 2026 | Static tiers, grouped expert execution, and graph residency for Apple NPUs | MoE experts/graphs | Apple Silicon NPU | Shape-specific NPU constraints and small-kernel dispatch/synchronization overhead | Different workload and platform; no RKNN sparse-SR evidence |

## Primary Sources Checked

- MobiSR: https://doi.org/10.1145/3300061.3345455
- SRNPU: https://doi.org/10.1109/JETCAS.2020.3014454
- SMSR: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Exploring_Sparsity_in_Image_Super-Resolution_for_Efficient_Inference_CVPR_2021_paper.html
- NAWQ-SR: https://doi.org/10.1109/TMC.2023.3255822
- Selective SR: https://doi.org/10.1609/aaai.v38i7.28560
- TileSR: https://doi.org/10.1109/INFOCOM52122.2024.10621208
- FYE-SR: https://doi.org/10.1145/3636534.3690698
- FAST: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Zhang_FAST_A_Framework_CVPR_2017_paper.html
- ESSR: https://doi.org/10.1109/TCSI.2024.3349581
- SkipSR: https://arxiv.org/abs/2510.08799
- EOS: https://doi.org/10.1145/3680207.3765252
- SCNN: https://doi.org/10.1145/3079856.3080254
- Eyeriss v2: https://doi.org/10.1109/JETCAS.2019.2910232
- PREMA: https://doi.org/10.1109/HPCA47549.2020.00027
- ART: https://doi.org/10.1145/3716368.3735215
- Hardware-aware DSA graph scheduling:
  https://www.usenix.org/conference/osdi23/presentation/zhao
- NPUMoE: https://arxiv.org/abs/2604.18788

## Novelty Threat Assessment

### High-risk objections

1. **Shared-substrate disclosure.** Paper 3 must disclose that its checkpoint,
   traces, and masks may also support related studies. Its live manuscript and
   package now import no Paper 2 result, denominator, counter, or mechanism
   measurement, so the scientific claims do not depend on Paper 2's disposition.
2. **Narrow workload breadth.** Two residual upscalers at one scale factor and
   one synthetic paired input still limit claims beyond this interface case.
3. **No physical energy.** The paper may discuss traffic and service time, but
   it cannot claim energy efficiency or low-power superiority.
4. **Missed real-time target.** Neither sparse 720p path reaches 30 FPS. The
   result must remain a boundary study, not a deployable real-time claim.
5. **Rejected selector.** Fixed and frozen masks are legitimate architecture
   workloads, but they do not demonstrate a generalized admission policy.

### Moderate-risk objections

1. Backend arithmetic and pixels are not identical, so absolute NPU-versus-FPGA
   quality ranking would be invalid.
2. Fifteen masks across five Godot sequences support geometry sensitivity but
   are not a broad benchmark suite.
3. Reviewers may view the six requirements as obvious unless every requirement
   is tied to a measured failure or controlled baseline.

## Search Conclusion

No checked primary source reports the same preregistered equal-count/equal-row
fragmentation comparison through RKNN, followed by the same 15-mask launch--
padding--traffic--host-tail--deadline accounting and transfer across two graph
cells and two Rockchip NPU targets. This is a bounded two-graph/two-device
characterization, not a broad new accelerator paradigm.
The literature contains multiple prior interface remedies; novelty rests on
the frozen physical measurements and NPU-local accounting, not on inventing them.

## Mandatory Pre-Submission Refresh

Repeat this search within two weeks of submission across SAMOS, IISWC, DATE,
FCCM, FPGA, ASP-DAC, MICRO, HPCA, ISCA, ASPLOS, IEEE Xplore, ACM DL, arXiv, and
Google Patents. Search the terms `sparse super-resolution NPU`, `selective SR
accelerator`, `output-mask dependency closure`, `persistent sparse CNN state`,
`deadline-safe neural refinement`, `tile compaction NPU`, and `atomic residual
commit`. This document is a research review, not a patent freedom-to-operate
opinion.
