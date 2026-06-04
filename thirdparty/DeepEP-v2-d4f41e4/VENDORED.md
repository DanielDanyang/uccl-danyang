# Vendored DeepEP V2 (in-tree copy, not a git submodule)

This directory is a **vendored copy** of DeepSeek DeepEP, committed directly into
the UCCL repo so we can patch it freely. It is intentionally **not** a git
submodule anymore (the submodule form made our edits fragile — they lived only in
the submodule working tree and were never captured by the parent repo history).

## Provenance

- Upstream: https://github.com/deepseek-ai/DeepEP
- Pinned commit: `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb` (`v1.2.1-32-gd4f41e4`)
- Nested dep `third-party/fmt` is also vendored in-tree (fmtlib/fmt), header path
  `third-party/fmt/include` (used by the V2 JIT bridge build). Upstream `figures/`
  (README images) is intentionally not vendored.

## Local modifications vs upstream

Recorded in `../DeepEP-v2-d4f41e4.local-changes.patch` (diff against pristine
`d4f41e4`). Summary:

| File | Change | Why |
|------|--------|-----|
| `csrc/elastic/buffer.hpp` | add `ElasticBuffer::get_native_v2_resources()` (+pybind) | export window/dev_comm pointers the native UCCL EFA wrapper needs |
| `csrc/kernels/backend/api.cuh` | add `NCCLSymmetricMemoryContext::get_raw_window_ptr()` | `raw_window_ptr` is private; needed as the EFA MR base |
| `csrc/jit/compiler.hpp` | JIT bridge tweak | UCCL native V2 JIT path |
| `csrc/jit/kernel_runtime.hpp` | JIT bridge tweak | UCCL native V2 JIT path |

## Updating from upstream

There is no `git submodule update`. To bump DeepEP:
1. Fetch the new upstream tree at the desired commit into a scratch checkout.
2. Re-copy it over this directory (keep `VENDORED.md`).
3. Re-apply the local edits (see the patch above) and update this file's commit hash
   + change table. Keep the edits minimal so re-applying stays cheap.

Paths are referenced by `ep/Makefile` (`DEEPEP_V2_ROOT`) and the Python wrapper
`ep/deep_ep_v2_wrapper/deep_ep/__init__.py`; keep this directory name stable.
