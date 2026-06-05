# CLAUDE.md

Project conventions and development rules for this repo live in **AGENTS.md** —
read and follow them:

@AGENTS.md

## Current focus

Native DeepEP V2 on AWS EFA, under `ep/`. Active branch for the rewrite:
`uccl-gin` (build a `handle::UCCLGin` GIN-shaped backend; the EFA `Rail` path goes
through UCCL D2H + proxy, `Lsa` forwards to NCCL/NVLink).

Key docs (read before working in `ep/`):
- `ep/docs/uccl_gin_plan.md` — the UCCL-GIN plan (architecture, file structure,
  phases, thirdparty minimal-patch strategy). **Authoritative going forward.**
- `worklog.md` — running log of experiments, results, and decisions; keep updating it.

## Repo facts that bite if forgotten

- DeepEP V2 is a **vendored in-tree copy**, not a git submodule:
  `thirdparty/DeepEP-v2-d4f41e4/` — edit it directly. See its `VENDORED.md`
  (upstream commit `d4f41e4`, local change list, re-vendor steps).
- `nccl/` (reference clone) and the paper PDF are git-ignored; don't commit them.
- The deleted fork-based native V2 path is preserved in git history only; do not
  restore `ep/include/v2_efa`, `ep/src/v2_efa_*.cc`, or `ep/deep_ep_v2_wrapper`.
