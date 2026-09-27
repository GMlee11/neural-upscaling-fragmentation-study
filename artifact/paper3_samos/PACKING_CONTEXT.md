# Historical packing screen: contextual inclusion

September 25, 2026. The earlier `paper3_packing_v1/E002` engineering screen is
included as background for the later reconstruction-placement study. It was run
on **RK3566**, not the later confirmation's RK3576. These are separate experiments.

The unchanged historical `docs/paper3_packing_v1/E002_result.md` says that the
screen had not been inserted into the reviewer artifact at that time. This note
records its subsequent contextual inclusion; it does not rewrite that history
or promote the screen to a confirmed performance result.

Run from the reviewer artifact root:

```powershell
python -B -m artifact.paper3_samos.replay_packing_context --root results/paper3_strengthening_v2/public_v1/retained_context/packing
```

For the original project layout, use `--root .`. An optional `--output` writes
the complete regenerated summary to a new JSON file; existing outputs are never
overwritten. Both layouts preserve the original `output/`, `results/` and `docs/`
namespaces inside the selected context root.

The replay checks the original seven-file source closure, unchanged report,
exact journal hashes, all eight development masks and eleven organizations per
graph, their deterministic execution order, and every policy mean and per-mask
result. Grouping has the lowest observed fixed-policy average in both graphs;
packing still wins on some individual masks. The hindsight combinations are
oracles, not deployable policies.

There is **one observation per organization/mask**, with conservative separate
exact calls for true image-boundary tiles. No stable ranking, general packing
limitation, or new algorithm is established. Raw tensors remain omitted:
recorded numerical checks and hash-bound prior audits are retained provenance,
not independent pixel revalidation by this compact utility. Initialization and
warmup remain outside the recorded complete-service measurement.
