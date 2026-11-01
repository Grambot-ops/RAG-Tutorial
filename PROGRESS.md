# Progress

The full instructions and code for every step are in `GUIDE.md`.

**Status:** `[ ]` todo · `[~]` in progress · `[x]` done · `[r]` done and reviewed

## Now

- **Current step:** 0.1 Workspace (Cargo.toml files, .gitignore)
- **Next:** 0.2 The model container
- **Blocked on:** nothing

## Chapter 00: Setup and corpus

- [ ] 0.1 Workspace: the three `Cargo.toml` files, `.gitignore`; `cargo build` passes
- [ ] 0.2 Model downloaded to `models/`; `compose.yaml` with `llm`; curl answers "Hoi"; ~8.4 GB VRAM
- [ ] 0.3 `config.rs`
- [ ] 0.4 PDFs downloaded; `corpus/acme/policy.typ` written and built into `D5-acme.pdf`
- [ ] 0.5 `corpus/corpus.toml` + `corpus.rs`
- [ ] 0.6 pdfium downloaded; `prepare.rs`, `lib.rs`, `main.rs`; 5 tests pass; `rag prepare` makes 416 pages
- [ ] 0.7 `corpus/golden.jsonl` with the first 10 questions

## Chapter 01: Long-context RAG

- [ ] 1.1 `llm.rs`
- [ ] 1.2 `longctx.rs`
- [ ] 1.3 `lib.rs` + `main.rs` with `ask`; 12 tests pass; cache hit on the 2nd question; medium refused

## Chapter 02: Vector RAG

- [ ] 2.0 bge-m3 + reranker downloaded; `qdrant`, `embed`, `rerank` in compose; curl checks pass
- [ ] 2.1 `chunk.rs`
- [ ] 2.2 `embed.rs`
- [ ] 2.3 `bm25.rs`
- [ ] 2.4 `store.rs`
- [ ] 2.5 `fusion.rs`
- [ ] 2.6 `rerank.rs`
- [ ] 2.7 `vector.rs` + final `lib.rs`/`main.rs`; 26 tests pass; `rag index`, `rag search`, `rag ask --mode vector` work

## Chapter 03: Multimodal RAG (code coming)

- [ ] 3.1 `textify.rs`: the scan question goes from NOT FOUND to 30.7
- [ ] 3.2 `colpali-server`
- [ ] 3.3 `maxsim.rs`
- [ ] 3.4 `pages` multivector collection
- [ ] 3.5 `multimodal.rs`

## Chapter 04: Compare, package, explain (code coming)

- [ ] 4.1 40 golden questions
- [ ] 4.2 `rag eval`
- [ ] 4.3 App image, `app` service, `rag serve`
- [ ] 4.4 README
- [ ] 4.5 Containers down, GPU free

---

## Your measurements

Fill these in with what *your* run prints. The guide's numbers are in brackets.

### Long-context (1.3)

| Question | Time | Prompt tokens | Cached | Hit/miss |
|---|---|---|---|---|
| 1st (Acme) | (15.1 s) | (17,447) | (0) | (miss) |
| 2nd (Dutch art. 4) | (5.1 s) | (17,445) | (17,413) | (hit) |
| 3rd (FTTP) | (0.2 s) | | | (hit, NOT FOUND) |

### Vector retrieval (2.7)

| Question | Dense top-1 | Sparse top-1 | Hybrid top-1 | Rerank top-1 |
|---|---|---|---|---|
| Dutch AI-geletterdheid | (D1-en p.51) | (D1-nl p.51) | (D1-nl p.51) | (D1-nl p.51) |
| EUR 12,500 threshold | | | | |

### Timings per vector question (2.7)

- retrieve: (0.03 s) · rerank: (~5.5 s) · generate: (1–4 s)

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Start from scratch; type all code from the guide | learn by doing |
| 2026-11-01 | Qwen3-VL-8B Q4 on llama.cpp instead of Strata 35B | the 35B pinned ~20 GB RAM and the PC ran out of memory; the 8B fits on the GPU, reads images, and caches simply |

## Review notes

<!-- feedback per step goes here -->
