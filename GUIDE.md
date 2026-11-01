# RAG Three Ways: a hands-on tutorial in Rust

You build three kinds of RAG in Rust, run them in Docker, and measure when each one wins:

1. **Long-context (CAG).** The whole corpus goes into the prompt, and the model server keeps it pre-read between questions.
2. **Vector RAG.** Chunk the text, embed it, search Qdrant by meaning and by keywords, rerank, answer.
3. **Multimodal RAG.** Find page images with ColPali, then answer from the images with a vision model.

## How to use this guide

- **Type the code yourself.** Every file is shown in full, exactly as it was compiled, tested and run on this PC. Typing it (not copy-pasting) is how it sticks. Read the comments as you go; they explain *why*.
- Work top to bottom. Every step has an ID (**1.2**) and ends with a **Check**: a command and what you should see.
- When a check passes: tick the step in `PROGRESS.md`, then `git add -A && git commit -m "1.2 longctx"`.
- If a check fails, compare your file with the guide line by line first. Still stuck? Ask for a review ("review 1.2").
- **Status:** chapters 00–02 are complete and verified. Chapters 03–04 are outlined; their code will be added after it has been verified the same way.

## Contents

- [Before you start](#before-you-start)
- [Folder structure](#folder-structure)
- [Chapter 00: Setup and corpus](#chapter-00-setup-and-corpus)
- [Chapter 01: Long-context RAG](#chapter-01-long-context-rag)
- [Chapter 02: Vector RAG](#chapter-02-vector-rag)
- [Chapter 03: Multimodal RAG (outline)](#chapter-03-multimodal-rag-outline)
- [Chapter 04: Compare, package, explain (outline)](#chapter-04-compare-package-explain-outline)
- [Lessons learned while building this](#lessons-learned-while-building-this)

---

## Before you start

### The models (small, so the PC doesn't run out of memory)

| Job | Model | Runs on | Memory |
|---|---|---|---|
| Answers, long-context, **and** reading images | **Qwen3-VL-8B-Instruct**, Q4_K_M (5.0 GB) + its vision projector (0.75 GB) | llama.cpp server, **GPU** | ~8.4 GB VRAM, ~0.4 GB RAM |
| Embeddings (ch02) | bge-m3, Q4_K_M (0.44 GB) | llama.cpp server, CPU | ~0.3 GB RAM |
| Reranker (ch02) | bge-reranker-v2-m3, Q4_K_M (0.44 GB) | llama.cpp server, CPU | ~0.3 GB RAM |
| Vector database (ch02) | Qdrant 1.20 | CPU | ~0.3 GB RAM |

Measured on this PC with everything running: **8.4 GB of 12 GB VRAM, about 1.3 GB RAM for all containers.**

Why not the 35B Strata model from `~/local-llm`? It keeps about 20 GB of its weights in pinned RAM. Together with Qdrant, the embedder, the reranker and `cargo` builds, that ran the 31 GB PC out of memory, and the machine froze. The 8B model fits entirely on the GPU, reads images too (so chapter 03 needs no second model), and makes the long-context cache simple.

### Rules

- **One GPU model at a time.** Nothing from `~/local-llm` may be running. Check with `nvidia-smi`: memory used should be under 1 GB before you start.
- **Never change `~/local-llm`.** This project has its own `models/` folder.
- **Docker only.** Build things so they *could* run on Kubernetes (config from env vars, health checks), but don't deploy Kubernetes.
- **Build with `--release`** for anything that does real work. A debug build of `rag prepare` took 4 minutes; the release build takes 7 seconds.

### Optional: remove leftovers from the old project

These containers, volumes and images are from the deleted first version. They're stopped and do nothing, but the old containers share the compose project name `rag` and will cause "orphan" warnings. Remove them if you like (this is permanent):

```bash
docker rm rag-lc-1 rag-vector-1 rag-mm-1 rag-colpali-1
docker volume rm rag_qdrant-storage rag_strata-data rag-strata-smoke-data
docker image rm rag/app:dev rag/colpali:dev rag/strata-vision:379f095
```

---

## Folder structure

This is what you'll have at the end of chapter 02:

```
~/Documents/rag/
├── GUIDE.md                  # this tutorial
├── PROGRESS.md               # your checklist + measurements
├── Cargo.toml                # workspace root                      (0.1)
├── .gitignore                #                                       (0.1)
├── compose.yaml              # all containers                       (0.2, 2.0)
├── models/                   # GGUF model files (git-ignored)      (0.2, 2.0)
├── vendor/pdfium/            # libpdfium.so (git-ignored)           (0.6)
├── corpus/
│   ├── corpus.toml           # documents + subsets                  (0.5)
│   ├── golden.jsonl          # test questions                       (0.7)
│   ├── acme/policy.typ       # the fictional policy                 (0.4)
│   ├── raw/                  # downloaded PDFs (git-ignored)        (0.4)
│   └── pages/                # generated .txt + .png (git-ignored)  (0.6)
└── crates/
    ├── ragkit/               # library: ALL the RAG logic
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── config.rs     # 0.3
    │       ├── corpus.rs     # 0.5
    │       ├── prepare.rs    # 0.6
    │       ├── llm.rs        # 1.1
    │       ├── longctx.rs    # 1.2
    │       ├── chunk.rs      # 2.1
    │       ├── embed.rs      # 2.2
    │       ├── bm25.rs       # 2.3
    │       ├── store.rs      # 2.4
    │       ├── fusion.rs     # 2.5
    │       ├── rerank.rs     # 2.6
    │       └── vector.rs     # 2.7
    └── rag/                  # binary: the `rag` command line
        ├── Cargo.toml
        └── src/main.rs       # only parses arguments and prints
```

**The one rule:** logic goes in `ragkit`, never in `main.rs`. Then every piece can be tested with `cargo test -p ragkit`, with no command line and no containers.

---

## Chapter 00: Setup and corpus

**Goal:** a workspace that builds, the model answering on port 8080, and the corpus as one text file and one image per page.

### 0.1 Workspace

You already ran `git init` and created `crates/ragkit` and `crates/rag`. Now replace the three `Cargo.toml` files with these. They list **every** crate the project will ever need, so you never have to come back to them.

**`Cargo.toml`** (the workspace root):

```toml
[workspace]
members = ["crates/*"]
resolver = "3"

[workspace.package]
edition = "2024"
version = "0.1.0"

[workspace.dependencies]
anyhow = "1"
axum = "0.8"
base64 = "0.23"
clap = { version = "4", features = ["derive"] }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
imageproc = { version = "0.27", default-features = false }
pdfium-render = "0.9"
qdrant-client = "1.20"
reqwest = { version = "0.13", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tempfile = "3"
tokio = { version = "1", features = ["full"] }
toml = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
wiremock = "0.6"
```

`[workspace.dependencies]` fixes each version in one place. The crates below then just say `tokio.workspace = true`.

**`crates/ragkit/Cargo.toml`**:

```toml
[package]
name = "ragkit"
edition.workspace = true
version.workspace = true

[dependencies]
anyhow.workspace = true
base64.workspace = true
image.workspace = true
imageproc.workspace = true
pdfium-render.workspace = true
qdrant-client.workspace = true
reqwest.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
toml.workspace = true
tracing.workspace = true

[dev-dependencies]
tempfile.workspace = true
wiremock.workspace = true
```

**`crates/rag/Cargo.toml`**:

```toml
[package]
name = "rag"
edition.workspace = true
version.workspace = true

[dependencies]
ragkit = { path = "../ragkit" }
anyhow.workspace = true
axum.workspace = true
clap.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

**`.gitignore`**: keep what Cargo wrote and make sure these lines are in it:

```gitignore
/target
/models/
/vendor/
/corpus/raw/
/corpus/pages/
/results/
```

`crates/ragkit/src/lib.rs` and `crates/rag/src/main.rs` get written in the next steps. For now leave `lib.rs` empty and `main.rs` as `fn main() {}`.

**Check:** `cargo build` finishes (the first build downloads and compiles all crates and takes a few minutes).

### 0.2 The model container

Download the model (5.8 GB in total) into the project's `models/` folder:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF \
  Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf --local-dir models
```

Create **`compose.yaml`**:

```yaml
name: rag

x-gpu: &gpu
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  # The language model: text AND images (Qwen3-VL-8B, Q4), fully on the GPU.
  llm:
    <<: *gpu
    image: ghcr.io/ggml-org/llama.cpp:server-cuda13
    container_name: rag-llm
    command: >
      -m /models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
      --mmproj /models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
      -c 24576 -np 1 -ngl 99 --flash-attn on -ctk q8_0 -ctv q8_0
      --host 0.0.0.0 --port 8080
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./models:/models:ro
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://127.0.0.1:8080/health"]
      interval: 5s
      retries: 60
```

What the llama-server flags mean:

| Flag | Meaning |
|---|---|
| `-m` / `--mmproj` | the language model, and the vision projector that turns images into tokens |
| `-c 24576` | context of 24,576 tokens (the small corpus is ~17,500) |
| `-np 1` | one slot: one request at a time. The slot keeps the last prompt cached (chapter 01) |
| `-ngl 99` | put all layers on the GPU |
| `--flash-attn on -ctk q8_0 -ctv q8_0` | store the KV cache in 8 bits: half the VRAM of the default |

**Check:**

```bash
docker compose up -d --wait llm
nvidia-smi --query-gpu=memory.used --format=csv     # about 8,400 MiB
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in Dutch, one word."}],"max_tokens":20}'
```

The answer is `"content":"Hoi"`. Look at the `usage` block too: `prompt_tokens_details.cached_tokens` will matter in chapter 01.

### 0.3 Configuration (`config.rs`)

Every URL and path, read from environment variables. On the host the defaults point at `127.0.0.1`; in a container, compose sets e.g. `LLM_URL=http://llm:8080/v1`. Same binary, different environment: that's what makes it Kubernetes-ready.

**`crates/ragkit/src/config.rs`**:

```rust
//! Step 0.3: every URL and path in one place, read from environment variables.
//!
//! On the host the defaults point at 127.0.0.1. In a container, compose sets e.g. LLM_URL=http://llm:8080/v1.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_url: String,
    pub llm_model: String,
    pub embed_url: String,
    pub rerank_url: String,
    pub qdrant_url: String,
    pub colpali_url: String,
    pub corpus_dir: PathBuf,
    pub pdfium_dir: PathBuf,
}

impl Config {
    /// Read the real environment.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Read settings through any lookup function, so tests don't touch the real environment.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let var = |key: &str, default: &str| get(key).unwrap_or_else(|| default.to_string());
        Config {
            llm_url: var("LLM_URL", "http://127.0.0.1:8080/v1"),
            llm_model: var("LLM_MODEL", "qwen3-vl-8b"),
            embed_url: var("EMBED_URL", "http://127.0.0.1:8081"),
            rerank_url: var("RERANK_URL", "http://127.0.0.1:8082"),
            qdrant_url: var("QDRANT_URL", "http://127.0.0.1:6334"),
            colpali_url: var("COLPALI_URL", "http://127.0.0.1:8083"),
            corpus_dir: PathBuf::from(var("CORPUS_DIR", "corpus")),
            pdfium_dir: PathBuf::from(var("PDFIUM_DIR", "vendor/pdfium/lib")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_localhost() {
        let cfg = Config::from_lookup(|_| None);
        assert_eq!(cfg.llm_url, "http://127.0.0.1:8080/v1");
        assert_eq!(cfg.qdrant_url, "http://127.0.0.1:6334");
    }

    #[test]
    fn env_overrides_default() {
        let cfg = Config::from_lookup(|key| (key == "LLM_URL").then(|| "http://llm:8080/v1".to_string()));
        assert_eq!(cfg.llm_url, "http://llm:8080/v1");
        assert_eq!(cfg.embed_url, "http://127.0.0.1:8081");
    }
}
```

Why `from_lookup`? Tests run in parallel and share the process environment. Passing a lookup function lets the tests fake the environment without touching it.

The tests can't run until `lib.rs` mentions the module. That happens in 0.6, so the check for this step is there.

### 0.4 The documents

| ID | Document | Why it's in the corpus |
|---|---|---|
| D1-en, D1-nl | AI Act, Regulation (EU) 2024/1689, in English and Dutch (144 pages each) | long, and in two languages: cross-language search |
| D2-eurostat | Eurostat "Key figures on Europe", 2026 edition (84 pages) | numbers and charts |
| D3-dd-belgium | Digital Decade 2025 country report: Belgium (41 pages) | tables, charts, exact values |
| D4-scan | a fake scan of D3's KPI table (made in 0.6) | no text layer: only vision can read it |
| D5-acme | a fictional company policy (2 pages) | the model *can't* know it, so a right answer proves retrieval worked |

Download them (EUR-Lex blocks scripts, so D1 comes from the EU Publications Office, which returns the PDF when you ask for it with an `Accept` header):

```bash
mkdir -p corpus/raw corpus/acme && cd corpus/raw
curl -sL -H "Accept: application/pdf" -H "Accept-Language: eng" -o D1-en.pdf "http://publications.europa.eu/resource/celex/32024R1689"
curl -sL -H "Accept: application/pdf" -H "Accept-Language: nld" -o D1-nl.pdf "http://publications.europa.eu/resource/celex/32024R1689"
curl -sL -o D2-eurostat.pdf "https://www.drugsandalcohol.ie/46268/1/key-figures-on-europe.pdf"
curl -sL -o D3-dd-belgium.pdf "https://data.consilium.europa.eu/doc/document/ST-10407-2025-ADD-34/en/pdf"
cd ../..
```

Now write the fictional policy. **`corpus/acme/policy.typ`**:

```typst
// Fictional company policy for the RAG tutorial. Every fact here is invented,
// so the model cannot know it from training: a correct answer proves retrieval worked.
#set page(paper: "a4", margin: 2.2cm)
#set text(font: "Libertinus Serif", size: 11pt, lang: "en")
#set heading(numbering: "1.")

#align(center)[
  #text(18pt, weight: "bold")[Acme Logistics NV]\
  #text(14pt)[Policy on the Use of Artificial Intelligence]\
  Version 2.3 · approved by the Board on 14 March 2026 · owner: the AI Steward
]

= Purpose and scope
This policy applies to every employee, contractor and intern of Acme Logistics NV and its subsidiaries in Belgium,
the Netherlands and Luxembourg. It covers any AI system that the company builds, buys or uses, including
general-purpose chatbots used through a web browser.

= Roles
The *AI Steward* is the person accountable for this policy. The current AI Steward is Ms. Lotte Verbeke, who reports
directly to the Chief Operating Officer. Each department appoints one *AI Champion* who keeps the department's
register of AI systems up to date.

The *AI Review Board* has five members: the AI Steward, the Data Protection Officer, the Chief Information Security
Officer, one representative of the works council and one rotating AI Champion. It meets every second Tuesday of the
month.

= Approval of new AI systems
Before a new AI system is used, the requesting department submits an AI Use Request to the AI Steward.

#table(
  columns: (auto, auto, auto),
  align: left,
  [*Expected yearly cost*], [*Who approves*], [*Maximum review time*],
  [up to EUR 12,500], [the AI Champion of the department], [5 working days],
  [EUR 12,500 to EUR 100,000], [the AI Steward], [14 calendar days],
  [above EUR 100,000], [the AI Review Board], [30 calendar days],
)

Any system that the EU AI Act classifies as high-risk always goes to the AI Review Board, whatever its cost.

= Rules for everyday use
+ Never paste customer personal data, salary data or unpublished financial results into an external AI tool.
+ AI-generated text that is sent to a customer must be read and approved by a human first.
+ Employees must label internal documents that were mostly written by AI with the tag "AI-assisted".
+ Training on approved AI tools is mandatory: at least 4 hours per employee per calendar year.

= AI literacy
Every new employee completes the course "AI at Acme" within 30 days of their start date. Managers complete an extra
2-hour module on the risks of automated decisions about staff.

Note: Acme's internal deadline for AI-literacy training for existing staff was 1 September 2025, seven months after
the date on which the EU AI Act's AI-literacy obligation started to apply.

= Incidents
A serious AI incident (for example: a wrong automated decision that affects a customer, or a data leak through an AI
tool) must be reported to the AI Steward within 24 hours. The AI Steward informs the Board within 72 hours.

= Prohibited uses
Acme never uses AI for emotion recognition of employees, for social scoring, or for ranking job applicants without a
human recruiter reviewing every application.

= Review
This policy is reviewed every 12 months. The next review is due in March 2027.
```

Build it into a PDF with the Typst image (`--user` makes the file yours instead of root's):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/corpus:/w" -w /w \
  ghcr.io/typst/typst:latest compile acme/policy.typ raw/D5-acme.pdf
```

**Check:** `pdfinfo corpus/raw/*.pdf | grep Pages` shows 144, 144, 84, 41 and 2 pages.

### 0.5 The corpus file and the page loader (`corpus.rs`)

**`corpus/corpus.toml`** lists the documents, the scan and the subsets:

```toml
# The documents. `file` is relative to the corpus folder.
[[docs]]
id = "D1-en"
file = "raw/D1-en.pdf"
lang = "en"
title = "AI Act, Regulation (EU) 2024/1689 (English)"

[[docs]]
id = "D1-nl"
file = "raw/D1-nl.pdf"
lang = "nl"
title = "AI-verordening, Verordening (EU) 2024/1689 (Nederlands)"

[[docs]]
id = "D2-eurostat"
file = "raw/D2-eurostat.pdf"
lang = "en"
title = "Eurostat, Key figures on Europe, 2026 edition"

[[docs]]
id = "D3-dd-belgium"
file = "raw/D3-dd-belgium.pdf"
lang = "en"
title = "Digital Decade 2025 country report: Belgium"

[[docs]]
id = "D5-acme"
file = "raw/D5-acme.pdf"
lang = "en"
title = "Acme Logistics NV, Policy on the Use of AI (fictional)"

# The fake scan (step 0.7): D3's KPI table page, turned into an image-only page.
[scan]
id = "D4-scan"
from_doc = "D3-dd-belgium"
from_page = 5
lang = "en"

# Page lists per subset ("full" = everything, no list needed).
# small leaves out D3 pages 5 and 11 on purpose: then the KPI values exist ONLY in the scan (chapter 03).
[subsets.small]
D1-en = "1,46,51-52,115,123"
D1-nl = "51-52,115,123"
D2-eurostat = "12,24,27,34-36"
D3-dd-belgium = "6-7,9-10,16-17"
D4-scan = "1"
D5-acme = "1-2"

[subsets.medium]
D1-en = "1-70"
D1-nl = "45-60,110-125"
D2-eurostat = "11-36"
D3-dd-belgium = "1-41"
D4-scan = "1"
D5-acme = "1-2"
```

| Subset | Pages | Tokens | What it shows |
|---|---|---|---|
| small | 25 | ~17,500 | fits long-context; fast experiments |
| medium | ~170 | ~180,000 (estimated) | far too big for a 24K context: where long-context stops |
| full | 416 | | everything |

Page numbers are **PDF page numbers** (the first page of the file is 1), not the numbers printed on the page. Eurostat's PDF page 24 is printed as "22". Citations use PDF page numbers, so you can always check them.

**`crates/ragkit/src/corpus.rs`**:

```rust
//! Step 0.8: the corpus as a list of pages.
//!
//! `corpus/corpus.toml` lists the documents and the subsets. `rag prepare` writes one text file and one image per
//! page plus `corpus/pages/manifest.jsonl`. This module reads them back.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// One document from `corpus.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct DocInfo {
    pub id: String,
    pub file: String,
    pub lang: String,
    pub title: String,
}

/// The fake scan: which rendered page to turn into an image-only "scanned" page.
#[derive(Debug, Clone, Deserialize)]
pub struct ScanInfo {
    pub id: String,
    pub from_doc: String,
    pub from_page: u32,
    pub lang: String,
}

/// The whole `corpus.toml`.
#[derive(Debug, Deserialize)]
pub struct CorpusFile {
    pub docs: Vec<DocInfo>,
    pub scan: Option<ScanInfo>,
    /// subset name -> (doc id -> page list such as "1-3,7")
    #[serde(default)]
    pub subsets: BTreeMap<String, BTreeMap<String, String>>,
}

impl CorpusFile {
    pub fn load(corpus_dir: &Path) -> Result<Self> {
        let path = corpus_dir.join("corpus.toml");
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// One line of `manifest.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PageMeta {
    pub doc: String,
    pub lang: String,
    pub page: u32,
    pub chars: usize,
    pub has_text: bool,
}

/// A page ready to use: its text and the path of its image.
#[derive(Debug, Clone)]
pub struct Page {
    pub doc: String,
    pub lang: String,
    pub page: u32,
    pub text: String,
    pub image: PathBuf,
    pub has_text: bool,
}

impl Page {
    /// How answers cite this page: "D3-dd-belgium p.16".
    pub fn cite(&self) -> String {
        format!("{} p.{}", self.doc, self.page)
    }
}

pub fn page_dir(corpus_dir: &Path, doc: &str) -> PathBuf {
    corpus_dir.join("pages").join(doc)
}
pub fn text_path(corpus_dir: &Path, doc: &str, page: u32) -> PathBuf {
    page_dir(corpus_dir, doc).join(format!("p{page:03}.txt"))
}
pub fn image_path(corpus_dir: &Path, doc: &str, page: u32) -> PathBuf {
    page_dir(corpus_dir, doc).join(format!("p{page:03}.png"))
}
pub fn manifest_path(corpus_dir: &Path) -> PathBuf {
    corpus_dir.join("pages").join("manifest.jsonl")
}

/// "1-3,7" -> [1, 2, 3, 7]
pub fn parse_ranges(spec: &str) -> Result<Vec<u32>> {
    let mut pages = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (u32, u32) = (a.trim().parse()?, b.trim().parse()?);
                if a > b {
                    bail!("bad range {part}: start is after end");
                }
                pages.extend(a..=b);
            }
            None => pages.push(part.parse().with_context(|| format!("bad page number {part}"))?),
        }
    }
    Ok(pages)
}

pub fn read_manifest(corpus_dir: &Path) -> Result<Vec<PageMeta>> {
    let path = manifest_path(corpus_dir);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("reading {} (did you run `rag prepare`?)", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).with_context(|| format!("bad manifest line: {l}")))
        .collect()
}

/// Load the pages of a subset ("small", "medium", or "full" for everything), sorted by doc and page.
pub fn load_pages(corpus_dir: &Path, subset: &str) -> Result<Vec<Page>> {
    let wanted: Option<BTreeMap<String, Vec<u32>>> = if subset == "full" {
        None
    } else {
        let file = CorpusFile::load(corpus_dir)?;
        let spec = file.subsets.get(subset).with_context(|| format!("no subset `{subset}` in corpus.toml"))?;
        let mut map = BTreeMap::new();
        for (doc, ranges) in spec {
            map.insert(doc.clone(), parse_ranges(ranges)?);
        }
        Some(map)
    };

    let mut pages = Vec::new();
    for meta in read_manifest(corpus_dir)? {
        if let Some(w) = &wanted {
            if !w.get(&meta.doc).is_some_and(|ps| ps.contains(&meta.page)) {
                continue;
            }
        }
        let text = fs::read_to_string(text_path(corpus_dir, &meta.doc, meta.page))?;
        pages.push(Page {
            image: image_path(corpus_dir, &meta.doc, meta.page),
            doc: meta.doc,
            lang: meta.lang,
            page: meta.page,
            text,
            has_text: meta.has_text,
        });
    }
    pages.sort_by(|a, b| (&a.doc, a.page).cmp(&(&b.doc, b.page)));
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_parse() {
        assert_eq!(parse_ranges("1-3,7").unwrap(), vec![1, 2, 3, 7]);
        assert_eq!(parse_ranges(" 5 ").unwrap(), vec![5]);
        assert!(parse_ranges("4-2").is_err());
        assert!(parse_ranges("x").is_err());
    }

    #[test]
    fn load_pages_filters_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let c = dir.path();
        fs::write(c.join("corpus.toml"), "docs = []\n[subsets.small]\nB = \"2\"\nA = \"1\"\n").unwrap();
        let mut manifest = String::new();
        for (doc, page) in [("B", 1), ("B", 2), ("A", 1)] {
            fs::create_dir_all(page_dir(c, doc)).unwrap();
            fs::write(text_path(c, doc, page), format!("text {doc}{page}")).unwrap();
            let m = PageMeta { doc: doc.into(), lang: "en".into(), page, chars: 7, has_text: true };
            manifest += &(serde_json::to_string(&m).unwrap() + "\n");
        }
        fs::write(manifest_path(c), manifest).unwrap();

        let small = load_pages(c, "small").unwrap();
        let got: Vec<String> = small.iter().map(Page::cite).collect();
        assert_eq!(got, vec!["A p.1", "B p.2"]);
        assert_eq!(small[1].text, "text B2");
        assert_eq!(load_pages(c, "full").unwrap().len(), 3);
        assert!(load_pages(c, "nope").is_err());
    }
}
```

### 0.6 PDF to pages (`prepare.rs`), the fake scan, and the first CLI

Download pdfium (Google's PDF engine, which the `pdfium-render` crate calls):

```bash
mkdir -p vendor/pdfium
curl -sL https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz | tar xz -C vendor/pdfium
ls vendor/pdfium/lib      # libpdfium.so
```

**`crates/ragkit/src/prepare.rs`**:

```rust
//! Steps 0.5 and 0.7: PDF -> one text file and one image per page, plus the fake scan.

use crate::corpus::{self, CorpusFile, DocInfo, PageMeta, ScanInfo};
use anyhow::{Context, Result, anyhow};
use image::{ImageReader, Luma, codecs::jpeg::JpegEncoder};
use imageproc::{
    filter::gaussian_blur_f32,
    geometric_transformations::{Border, Interpolation, rotate_about_center},
    noise::gaussian_noise,
};
use pdfium_render::prelude::*;
use std::{fs, io::Cursor, path::Path};

/// Page images are rendered this wide (about 150 dpi for A4). Enough for ColPali and the vision model.
pub const PAGE_WIDTH: i32 = 1240;

/// A page counts as "has text" above this many characters. Scans have none.
pub const MIN_TEXT_CHARS: usize = 50;

/// Load libpdfium.so from a folder (vendor/pdfium/lib on the host).
pub fn bind_pdfium(dir: &Path) -> Result<Pdfium> {
    let path = Pdfium::pdfium_platform_library_name_at_path(dir);
    let bindings = Pdfium::bind_to_library(&path)
        .map_err(|e| anyhow!("cannot load {}: {e} (set PDFIUM_DIR)", path.display()))?;
    Ok(Pdfium::new(bindings))
}

/// Tidy extracted text: unify line endings, trim line ends, at most one blank line in a row.
pub fn clean_text(raw: &str) -> String {
    let mut out = String::new();
    let mut blank_run = 0;
    for line in raw.replace("\r\n", "\n").replace('\r', "\n").lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

/// Write every page of one PDF as pNNN.txt and pNNN.png. Returns the manifest lines.
pub fn prepare_doc(pdfium: &Pdfium, corpus_dir: &Path, doc: &DocInfo) -> Result<Vec<PageMeta>> {
    let pdf_path = corpus_dir.join(&doc.file);
    let pdf = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .map_err(|e| anyhow!("opening {}: {e}", pdf_path.display()))?;
    fs::create_dir_all(corpus::page_dir(corpus_dir, &doc.id))?;
    let render = PdfRenderConfig::new().set_target_width(PAGE_WIDTH);

    let mut metas = Vec::new();
    for (i, page) in pdf.pages().iter().enumerate() {
        let n = i as u32 + 1;
        let text = clean_text(&page.text().map(|t| t.all()).unwrap_or_default());
        fs::write(corpus::text_path(corpus_dir, &doc.id, n), &text)?;

        let img = page
            .render_with_config(&render)
            .and_then(|bitmap| bitmap.as_image())
            .map_err(|e| anyhow!("rendering {} page {n}: {e}", doc.id))?;
        img.into_rgb8().save(corpus::image_path(corpus_dir, &doc.id, n))?;

        let chars = text.chars().count();
        metas.push(PageMeta { doc: doc.id.clone(), lang: doc.lang.clone(), page: n, chars, has_text: chars > MIN_TEXT_CHARS });
    }
    Ok(metas)
}

/// Turn a rendered page into an image-only "scan": gray, tilted, noisy, blurred, JPEG-damaged, no text layer.
pub fn make_scan(corpus_dir: &Path, scan: &ScanInfo) -> Result<PageMeta> {
    let src = corpus::image_path(corpus_dir, &scan.from_doc, scan.from_page);
    let gray = ImageReader::open(&src).with_context(|| format!("opening {}", src.display()))?.decode()?.into_luma8();

    let tilted = rotate_about_center(&gray, 1.5_f32.to_radians(), Interpolation::Bilinear, Border::Constant(Luma([255u8])));
    let noisy = gaussian_noise(&tilted, 0.0, 12.0, 42);
    let soft = gaussian_blur_f32(&noisy, 0.8);

    // Round-trip through a low-quality JPEG for the typical compression blocks.
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 60).encode_image(&soft)?;
    let damaged = ImageReader::new(Cursor::new(jpeg)).with_guessed_format()?.decode()?;

    fs::create_dir_all(corpus::page_dir(corpus_dir, &scan.id))?;
    damaged.save(corpus::image_path(corpus_dir, &scan.id, 1))?;
    fs::write(corpus::text_path(corpus_dir, &scan.id, 1), "")?; // a scan has no text layer
    Ok(PageMeta { doc: scan.id.clone(), lang: scan.lang.clone(), page: 1, chars: 0, has_text: false })
}

/// `rag prepare`: every document in corpus.toml, then the scan, then the manifest.
pub fn prepare_all(pdfium: &Pdfium, corpus_dir: &Path) -> Result<Vec<PageMeta>> {
    let file = CorpusFile::load(corpus_dir)?;
    let mut all = Vec::new();
    for doc in &file.docs {
        let metas = prepare_doc(pdfium, corpus_dir, doc)?;
        tracing::info!(doc = %doc.id, pages = metas.len(), "prepared");
        all.extend(metas);
    }
    if let Some(scan) = &file.scan {
        all.push(make_scan(corpus_dir, scan)?);
        tracing::info!(doc = %scan.id, "scan made");
    }
    let lines: Vec<String> = all.iter().map(serde_json::to_string).collect::<Result<_, _>>()?;
    fs::write(corpus::manifest_path(corpus_dir), lines.join("\n") + "\n")?;
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_tidies() {
        let raw = "Title  \r\n\r\n\r\n\r\nBody line\t\nmore\n\n";
        assert_eq!(clean_text(raw), "Title\n\nBody line\nmore");
    }
}
```

**`crates/ragkit/src/lib.rs`** (it grows each chapter):

```rust
//! ragkit: all the RAG logic. The `rag` binary only parses arguments and calls into this crate.

pub mod config;
pub mod corpus;
pub mod prepare;
```

**`crates/rag/src/main.rs`** (first version):

```rust
//! `rag`: the command line. Only argument parsing and printing; the logic lives in ragkit.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ragkit::config::Config;
use ragkit::prepare;

#[derive(Parser)]
#[command(name = "rag", about = "RAG three ways: long-context, vector, multimodal")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// PDFs -> one .txt and one .png per page, plus the fake scan and the manifest
    Prepare,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cfg = Config::from_env();

    match Cli::parse().command {
        Command::Prepare => {
            let pdfium = prepare::bind_pdfium(&cfg.pdfium_dir)?;
            let pages = prepare::prepare_all(&pdfium, &cfg.corpus_dir)?;
            println!("prepared {} pages into {}/pages", pages.len(), cfg.corpus_dir.display());
        }
    }
    Ok(())
}
```

**Check:**

```bash
cargo test -p ragkit                       # 5 passed
cargo run --release -p rag -- prepare      # "prepared 416 pages into corpus/pages", ~7 s
```

Then open `corpus/pages/D4-scan/p001.png`: a grey, tilted, noisy copy of D3's KPI table. Open `corpus/pages/D5-acme/p001.txt` and compare it with the PDF.

### 0.7 The first golden questions

**`corpus/golden.jsonl`**: one JSON object per line. Each fact was checked against the page files.

```json
{"id":"q01","lang":"en","category":"lookup","question":"Who approves an AI system at Acme that costs EUR 50,000 per year, and how long may the review take?","answer":"The AI Steward, within 14 calendar days.","must_contain":["AI Steward","14"],"sources":["D5-acme p.1"]}
{"id":"q02","lang":"nl","category":"multilingual","question":"Wat moeten aanbieders en gebruiksverantwoordelijken doen volgens artikel 4 van de AI-verordening?","answer":"Maatregelen nemen voor een toereikend niveau van AI-geletterdheid bij hun personeel.","must_contain":["AI-geletterdheid"],"sources":["D1-nl p.51"]}
{"id":"q03","lang":"en","category":"cross-doc","question":"From which date do the AI Act's prohibitions apply, and how many months after that was Acme's AI-literacy deadline?","answer":"2 February 2025; Acme's deadline (1 September 2025) was seven months later.","must_contain":["2 February 2025"],"sources":["D1-en p.123","D5-acme p.1"]}
{"id":"q04","lang":"en","category":"visual","question":"What was Belgium's Fibre to the Premises (FTTP) coverage in DESI 2025?","answer":"30.7%","must_contain":["30.7"],"sources":["D4-scan p.1"]}
{"id":"q05","lang":"en","category":"unanswerable","question":"What was Acme Logistics' annual revenue in 2025?","answer":"NOT FOUND","must_contain":["NOT FOUND"],"sources":[]}
{"id":"q06","lang":"en","category":"numeric","question":"What is the budget of Belgium's revised Digital Decade roadmap?","answer":"EUR 913.71 million (0.15% of GDP), for 166 measures.","must_contain":["913.71"],"sources":["D3-dd-belgium p.7"]}
{"id":"q07","lang":"en","category":"numeric","question":"What was the EU employment rate for people aged 20 to 64 in 2025?","answer":"76.1%","must_contain":["76.1"],"sources":["D2-eurostat p.24"]}
{"id":"q08","lang":"en","category":"lookup","question":"What share of Belgian citizens say that digitalisation of daily services makes their lives easier?","answer":"72%","must_contain":["72"],"sources":["D3-dd-belgium p.6"]}
{"id":"q09","lang":"nl","category":"multilingual","question":"Wat is de maximale boete voor verboden AI-praktijken volgens de AI-verordening?","answer":"Tot 35 000 000 EUR of, voor een onderneming, tot 7 % van de wereldwijde jaaromzet.","must_contain":["35 000 000"],"sources":["D1-nl p.115"]}
{"id":"q10","lang":"en","category":"lookup","question":"How many members does Acme's AI Review Board have, and when does it meet?","answer":"Five members; every second Tuesday of the month.","must_contain":["second Tuesday"],"sources":["D5-acme p.1"]}
```

`must_contain` is what the eval in chapter 04 looks for in an answer. q04 can only be answered from the scan; q05 isn't in the corpus at all (the right answer is `NOT FOUND`).

**Chapter 00 is done.** Commit.

---

## Chapter 01: Long-context RAG

**Idea:** no search. Put the whole small corpus (~17,500 tokens) in the prompt and let the model read everything. That's simple, and good at questions that combine several documents. But it's limited by the context window (24K here), and reading 17,500 tokens takes about 15 seconds.

**The cache fixes the time.** After each request, llama-server keeps the computed prompt (the "KV cache") in its slot. If the next request starts with **exactly the same text**, it skips that part and reads only what's new. So: rules + corpus first (always identical), the question last (the only part that changes). First question: about 15 s. Every next question: about 3 s, mostly spent writing the answer.

### 1.1 The chat client (`llm.rs`)

llama-server speaks the OpenAI chat API: POST `/v1/chat/completions` with a list of messages.

**`crates/ragkit/src/llm.rs`**:

```rust
//! Step 1.1: a small client for an OpenAI-compatible chat API (llama.cpp's llama-server).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Message content: plain text, or a list of parts (text + images, used in chapter 03).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Message { role: Role::System, content: Content::Text(text.into()) }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Message { role: Role::User, content: Content::Text(text.into()) }
    }
    pub fn user_parts(parts: Vec<Part>) -> Self {
        Message { role: Role::User, content: Content::Parts(parts) }
    }
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for ChatOptions {
    fn default() -> Self {
        // temperature 0: the same question gives the same answer, which makes testing and eval fair
        ChatOptions { max_tokens: 1024, temperature: 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct ChatAnswer {
    pub text: String,
    pub prompt_tokens: u32,
    /// Prompt tokens the server did not have to read again (prefix cache).
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub elapsed: Duration,
}

// ---- wire format (only the fields we use) ----

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    max_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Clone)]
pub struct Llm {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl Llm {
    pub fn new(base_url: &str, model: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600)) // reading a long prompt takes a while
            .build()?;
        Ok(Llm { http, base_url: base_url.trim_end_matches('/').to_string(), model: model.to_string() })
    }

    pub async fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<ChatAnswer> {
        let started = Instant::now();
        let body = ChatRequest { model: &self.model, messages, max_tokens: opts.max_tokens, temperature: opts.temperature };
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self.http.post(&url).json(&body).send().await.with_context(|| format!("POST {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("LLM returned {status}: {text}");
        }
        let parsed: ChatResponse = resp.json().await.context("parsing chat response")?;
        let choice = parsed.choices.into_iter().next().context("chat response has no choices")?;

        Ok(ChatAnswer {
            text: choice.message.content.unwrap_or_default().trim().to_string(),
            prompt_tokens: parsed.usage.prompt_tokens,
            cached_tokens: parsed.usage.prompt_tokens_details.map(|d| d.cached_tokens).unwrap_or(0),
            completion_tokens: parsed.usage.completion_tokens,
            elapsed: started.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn image_parts_serialize_like_openai() {
        let m = Message::user_parts(vec![
            Part::ImageUrl { image_url: ImageUrl { url: "data:image/png;base64,AAAA".into() } },
            Part::Text { text: "What is this?".into() },
        ]);
        let v = serde_json::to_value(m).unwrap();
        assert_eq!(v["content"][0]["type"], "image_url");
        assert_eq!(v["content"][0]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(v["content"][1]["type"], "text");
        assert_eq!(serde_json::to_value(Message::user("hi")).unwrap()["content"], "hi");
    }

    #[tokio::test]
    async fn chat_parses_answer_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({"temperature": 0.0})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": " Hoi "}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 15}}
            })))
            .mount(&server)
            .await;

        let llm = Llm::new(&format!("{}/v1", server.uri()), "test").unwrap();
        let a = llm.chat(&[Message::user("hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(a.text, "Hoi");
        assert_eq!((a.prompt_tokens, a.cached_tokens, a.completion_tokens), (20, 15, 3));
    }

    #[tokio::test]
    async fn chat_error_includes_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("the request exceeds the available context size"))
            .mount(&server)
            .await;
        let llm = Llm::new(&server.uri(), "test").unwrap();
        let err = llm.chat(&[Message::user("hi")], &ChatOptions::default()).await.unwrap_err();
        assert!(err.to_string().contains("exceeds the available context"), "{err}");
    }
}
```

Things to notice:

- Request and response are plain serde structs with only the fields we need. `#[serde(default)]` keeps parsing working when a server leaves a field out.
- `Content` is `untagged`: plain text serializes as a string, a list of parts as an array. Images (chapter 03) use the parts form.
- A non-200 status becomes an error **that includes the server's message**. "Context size exceeded" is much easier to debug than "400".
- The tests use `wiremock`: a fake HTTP server, so they run without the model.

### 1.2 Long-context (`longctx.rs`)

**`crates/ragkit/src/longctx.rs`**:

```rust
//! Steps 1.2-1.3: long-context RAG (also called CAG, cache-augmented generation).
//!
//! The whole corpus goes into the system message. llama-server keeps the computed prompt (its KV cache) after each
//! request. When the next request starts with exactly the same text, it skips that part and only reads what is new:
//! the question. So the first question reads ~17K tokens (slow), every next question only ~30 (fast).

use crate::corpus::Page;
use crate::llm::{ChatAnswer, ChatOptions, Llm, Message};
use anyhow::{Result, bail};

/// llama-server runs with a 24,576-token context (compose.yaml). Our estimate overcounts by ~20%, so this
/// leaves room for the question and a 1,024-token answer.
pub const MAX_CORPUS_TOKENS: usize = 22_000;

pub const SYSTEM_RULES: &str = "You answer questions using only the documents you are given.
Rules:
- Use only facts from the documents. Do not use outside knowledge.
- Cite every fact with its source in square brackets, like [D3-dd-belgium p.16].
- Copy numbers exactly as they are written in the documents.
- Answer in the language of the question.
- If the documents do not contain the answer, reply exactly: NOT FOUND";

/// Rough token estimate: about 4 characters per token for EN/NL text; 3.5 keeps a safety margin.
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() as f64 / 3.5).ceil() as usize
}

/// All pages as tagged blocks. Must be byte-for-byte the same every time, or the cache can't be reused.
pub fn render_corpus(pages: &[Page]) -> String {
    let mut out = String::new();
    for p in pages {
        out.push_str(&format!("<page doc=\"{}\" page=\"{}\" lang=\"{}\">\n", p.doc, p.page, p.lang));
        out.push_str(p.text.trim());
        out.push_str("\n</page>\n");
    }
    out
}

/// Did the server reuse its cache? The question is tiny, so >= 90% of the prompt must have been cached.
pub fn is_cache_hit(a: &ChatAnswer) -> bool {
    a.prompt_tokens > 0 && a.cached_tokens as u64 * 10 >= a.prompt_tokens as u64 * 9
}

pub struct LongContext {
    llm: Llm,
    /// Rules + the whole corpus: the fixed start of every request.
    system: String,
}

impl LongContext {
    pub fn new(llm: Llm, pages: &[Page]) -> Result<Self> {
        let corpus = render_corpus(pages);
        let tokens = estimate_tokens(&corpus);
        if tokens > MAX_CORPUS_TOKENS {
            bail!(
                "corpus is about {tokens} tokens; long-context fits {MAX_CORPUS_TOKENS}. Use a smaller subset or vector RAG."
            );
        }
        Ok(LongContext { llm, system: format!("{SYSTEM_RULES}\n\nDocuments:\n{corpus}") })
    }

    /// The fixed system message first, the question last: only the question differs between requests.
    pub fn messages(&self, question: &str) -> Vec<Message> {
        vec![Message::system(&self.system), Message::user(question)]
    }

    pub async fn ask(&self, question: &str) -> Result<ChatAnswer> {
        self.llm.chat(&self.messages(question), &ChatOptions::default()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    fn page(doc: &str, n: u32, text: &str) -> Page {
        Page { doc: doc.into(), lang: "en".into(), page: n, text: text.into(), image: PathBuf::new(), has_text: true }
    }
    fn llm() -> Llm {
        Llm::new("http://127.0.0.1:9", "x").unwrap()
    }

    #[test]
    fn render_is_deterministic_and_tagged() {
        let pages = vec![page("D5-acme", 2, "Threshold is EUR 12,500."), page("D3", 16, "30.7 %")];
        let a = render_corpus(&pages);
        assert_eq!(a, render_corpus(&pages));
        assert!(a.contains("<page doc=\"D5-acme\" page=\"2\" lang=\"en\">"));
        assert!(a.contains("30.7 %"));
    }

    #[test]
    fn only_the_question_differs() {
        let lc = LongContext::new(llm(), &[page("D", 1, "text")]).unwrap();
        let (a, b) = (lc.messages("first?"), lc.messages("second?"));
        assert_eq!(a[0], b[0], "the system message (rules + corpus) must be identical");
        assert_ne!(a[1], b[1]);
    }

    #[test]
    fn too_big_corpus_is_refused() {
        let huge = vec![page("D", 1, &"word ".repeat(100_000))];
        assert!(LongContext::new(llm(), &huge).is_err());
    }

    #[test]
    fn cache_hit_rule() {
        let a = |prompt, cached| ChatAnswer {
            text: String::new(),
            prompt_tokens: prompt,
            cached_tokens: cached,
            completion_tokens: 0,
            elapsed: Duration::ZERO,
        };
        assert!(is_cache_hit(&a(14174, 14156)));
        assert!(!is_cache_hit(&a(14169, 1)));
        assert!(!is_cache_hit(&a(0, 0)));
    }
}
```

The test `only_the_question_differs` guards the cache: if anything in the system message changed between questions (a timestamp, a different page order), the server could never reuse it.

### 1.3 The CLI with `ask`

**`crates/ragkit/src/lib.rs`**:

```rust
//! ragkit: all the RAG logic. The `rag` binary only parses arguments and calls into this crate.

pub mod config;
pub mod corpus;
pub mod prepare;

pub mod llm;
pub mod longctx;
```

**`crates/rag/src/main.rs`** (replace the whole file):

```rust
//! `rag`: the command line. Only argument parsing and printing; the logic lives in ragkit.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use ragkit::config::Config;
use ragkit::{corpus, llm, longctx, prepare};
use std::io::Write;

#[derive(Parser)]
#[command(name = "rag", about = "RAG three ways: long-context, vector, multimodal")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// PDFs -> one .txt and one .png per page, plus the fake scan and the manifest
    Prepare,
    /// Ask a question. Without a question: read questions from stdin, one per line
    Ask {
        #[arg(long, value_enum, default_value = "lc")]
        mode: Mode,
        #[arg(long, default_value = "small")]
        subset: String,
        question: Option<String>,
    },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Mode {
    /// No documents at all (shows why RAG is needed)
    Plain,
    /// Long-context: the whole subset in the prompt
    Lc,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cfg = Config::from_env();

    match Cli::parse().command {
        Command::Prepare => {
            let pdfium = prepare::bind_pdfium(&cfg.pdfium_dir)?;
            let pages = prepare::prepare_all(&pdfium, &cfg.corpus_dir)?;
            println!("prepared {} pages into {}/pages", pages.len(), cfg.corpus_dir.display());
        }

        Command::Ask { mode, subset, question } => {
            let interactive = question.is_none();
            let mut one = question;
            let llm = llm::Llm::new(&cfg.llm_url, &cfg.llm_model)?;
            match mode {
                Mode::Plain => {
                    while let Some(q) = next_question(&mut one, interactive) {
                        let a = llm.chat(&[llm::Message::user(q)], &llm::ChatOptions::default()).await?;
                        println!("{}\n  [{:.1} s]", a.text, a.elapsed.as_secs_f64());
                    }
                }
                Mode::Lc => {
                    let pages = corpus::load_pages(&cfg.corpus_dir, &subset)?;
                    let lc = longctx::LongContext::new(llm, &pages)?;
                    while let Some(q) = next_question(&mut one, interactive) {
                        let a = lc.ask(&q).await?;
                        println!("{}", a.text);
                        println!(
                            "  [{:.1} s, {} prompt tokens, {} cached, cache {}]",
                            a.elapsed.as_secs_f64(),
                            a.prompt_tokens,
                            a.cached_tokens,
                            if longctx::is_cache_hit(&a) { "hit" } else { "miss" }
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// The next question: the one from the command line (once), or the next line typed on stdin.
fn next_question(one: &mut Option<String>, interactive: bool) -> Option<String> {
    if !interactive {
        return one.take();
    }
    loop {
        print!("question> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).ok()? == 0 {
            return None; // Ctrl-D
        }
        if !line.trim().is_empty() {
            return Some(line.trim().to_string());
        }
    }
}
```

**Check:**

```bash
cargo test -p ragkit                       # 12 passed
cargo build --release -p rag

# Plain: no documents. The model can't know the fictional Acme policy:
./target/release/rag ask --mode plain "Who approves an AI system at Acme that costs EUR 50,000 per year?"

# Long-context, interactive: type the questions one by one (Ctrl-D to stop)
./target/release/rag ask --mode lc
```

Ask these three, one after the other:

1. `Who approves an AI system at Acme that costs EUR 50,000 per year, and how long may the review take?`
2. `Wat moeten aanbieders en gebruiksverantwoordelijken doen volgens artikel 4 van de AI-verordening?`
3. `What was Belgium's FTTP coverage in DESI 2025?`

What this PC printed:

```
The AI Steward approves an AI system at Acme that costs EUR 50,000 per year [D5-acme p.1]. The review may take up to 14 calendar days [D5-acme p.1].
  [15.1 s, 17447 prompt tokens, 0 cached, cache miss]
Volgens artikel 4 van de AI-verordening moeten aanbieders en gebruiksverantwoordelijken maatregelen nemen om ... [D1-en p.51].
  [5.1 s, 17445 prompt tokens, 17413 cached, cache hit]
NOT FOUND
  [0.2 s, 17434 prompt tokens, 17413 cached, cache hit]
```

Three things to see:

- **Cache:** the first question reads everything (15 s). The next ones reuse 17,413 tokens. The cache lives in the server, so it even survives between separate `rag` runs, until another request (for example a vector-mode question) replaces it.
- **NOT FOUND for FTTP:** correct! The number only exists in the scan, which has no text. Chapter 03 fixes this.
- **A wrong citation:** the Dutch answer is right but cites `D1-en` instead of `D1-nl`. Small models are sloppier with citations. Keep an eye on it; the eval will count it.

Then try the limit: `./target/release/rag ask --mode lc --subset medium "test"` stops with `corpus is about 181804 tokens; long-context fits 22000`. That's the main weakness of long-context.

Write your own times in `PROGRESS.md`. **Commit.**

---

## Chapter 02: Vector RAG

**Idea:** don't read everything. Split the text into chunks and give each chunk two kinds of index: a **dense** vector (meaning, works across languages) and a **sparse** BM25 vector (exact words and numbers). Search both, merge the two rankings with **RRF**, let a **reranker** re-score the top 20 closely, and give only the best 5 chunks to the model. This scales to any corpus size and costs little per question, but it can miss answers that are spread over many places.

### 2.0 Three more containers

Download the two small models:

```bash
hf download gpustack/bge-m3-GGUF bge-m3-Q4_K_M.gguf --local-dir models
hf download gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-Q4_K_M.gguf --local-dir models
```

Replace **`compose.yaml`** with the full version:

```yaml
name: rag

x-gpu: &gpu
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  # The language model: text AND images (Qwen3-VL-8B, Q4), fully on the GPU.
  llm:
    <<: *gpu
    image: ghcr.io/ggml-org/llama.cpp:server-cuda13
    container_name: rag-llm
    command: >
      -m /models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
      --mmproj /models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
      -c 24576 -np 1 -ngl 99 --flash-attn on -ctk q8_0 -ctv q8_0
      --host 0.0.0.0 --port 8080
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./models:/models:ro
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://127.0.0.1:8080/health"]
      interval: 5s
      retries: 60

  qdrant:
    image: qdrant/qdrant:v1.20.0
    container_name: rag-qdrant
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - qdrant-storage:/qdrant/storage

  embed:
    image: ghcr.io/ggml-org/llama.cpp:server
    container_name: rag-embed
    command: >
      -m /models/bge-m3-Q4_K_M.gguf --embedding --pooling cls
      -c 8192 -b 2048 -ub 2048 -np 4 --host 0.0.0.0 --port 8080
    ports:
      - "127.0.0.1:8081:8080"
    volumes:
      - ./models:/models:ro

  rerank:
    image: ghcr.io/ggml-org/llama.cpp:server
    container_name: rag-rerank
    command: >
      -m /models/bge-reranker-v2-m3-Q4_K_M.gguf --rerank
      -c 8192 -b 2048 -ub 2048 -np 4 --host 0.0.0.0 --port 8080
    ports:
      - "127.0.0.1:8082:8080"
    volumes:
      - ./models:/models:ro

volumes:
  qdrant-storage:
```

The embedder and reranker run on the CPU, so they take nothing from the GPU. `-np 4` lets each handle 4 requests at once; `-ub 2048` must be at least as big as the longest text you send (our chunks are ~400 tokens).

**Check:**

```bash
docker compose up -d --wait
curl -s localhost:8081/v1/embeddings -H 'content-type: application/json' -d '{"input":["hallo wereld"]}' | head -c 120
curl -s localhost:8082/v1/rerank -H 'content-type: application/json' \
  -d '{"query":"capital of Belgium","documents":["Paris is the capital of France.","Brussels is the capital of Belgium."]}'
```

The rerank answer gives index 1 (Brussels) a high score (~9) and index 0 a negative one. The Qdrant dashboard is at http://127.0.0.1:6333/dashboard.

### 2.1 Chunking (`chunk.rs`)

**`crates/ragkit/src/chunk.rs`**:

```rust
//! Step 2.1: split pages into overlapping chunks of about `max_tokens` tokens.
//!
//! Paragraphs first, then sentences, and only as a last resort a hard cut. All lengths are counted in chars
//! (never bytes), so Dutch accents can't be cut in half.

use crate::corpus::Page;
use serde::{Deserialize, Serialize};

pub const MAX_TOKENS: usize = 400;
pub const OVERLAP_TOKENS: usize = 60;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: u64,
    pub doc: String,
    pub lang: String,
    pub page: u32,
    pub text: String,
}

impl Chunk {
    pub fn cite(&self) -> String {
        format!("{} p.{}", self.doc, self.page)
    }
}

/// FNV-1a: a tiny, stable hash (std's DefaultHasher may change between Rust versions).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Same doc/page/index -> same id, so indexing twice overwrites instead of duplicating.
/// Shifted right by one bit: Qdrant stores payload numbers as signed 64-bit, so ids must stay below 2^63.
pub fn chunk_id(doc: &str, page: u32, index: usize) -> u64 {
    fnv1a(format!("{doc}/{page}/{index}").as_bytes()) >> 1
}

/// The same estimate as long-context: ~3.5 chars per token.
fn tokens_to_chars(tokens: usize) -> usize {
    (tokens as f64 * 3.5) as usize
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn split_sentences(paragraph: &str) -> Vec<String> {
    let chars: Vec<char> = paragraph.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        let sentence_end = matches!(c, '.' | '?' | '!' | ';') && chars.get(i + 1).is_some_and(|n| n.is_whitespace());
        if sentence_end || c == '\n' {
            if !current.trim().is_empty() {
                out.push(current.trim().to_string());
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// Pieces no longer than `max_chars`: whole paragraphs, else sentences, else hard cuts.
fn split_pieces(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if char_len(para) <= max_chars {
            out.push(para.to_string());
            continue;
        }
        for sentence in split_sentences(para) {
            if char_len(&sentence) <= max_chars {
                out.push(sentence);
            } else {
                let chars: Vec<char> = sentence.chars().collect();
                out.extend(chars.chunks(max_chars).map(|w| w.iter().collect::<String>()));
            }
        }
    }
    out
}

/// The last ~`n` chars of `s`, starting at a word boundary.
fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        return s.to_string();
    }
    let t: String = chars[chars.len() - n..].iter().collect();
    match t.find(char::is_whitespace) {
        Some(i) => t[i..].trim_start().to_string(),
        None => t,
    }
}

pub fn chunk_page(page: &Page, max_tokens: usize, overlap_tokens: usize) -> Vec<Chunk> {
    let max_chars = tokens_to_chars(max_tokens);
    let overlap_chars = tokens_to_chars(overlap_tokens);

    let mut texts: Vec<String> = Vec::new();
    let mut current = String::new();
    // Pieces leave room for the overlap and a newline, so overlap + piece always fits.
    for piece in split_pieces(&page.text, max_chars - overlap_chars - 1) {
        if !current.is_empty() && char_len(&current) + 1 + char_len(&piece) > max_chars {
            let overlap = tail(&current, overlap_chars);
            texts.push(std::mem::replace(&mut current, overlap));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(&piece);
    }
    if !current.trim().is_empty() {
        texts.push(current);
    }

    texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            id: chunk_id(&page.doc, page.page, i),
            doc: page.doc.clone(),
            lang: page.lang.clone(),
            page: page.page,
            text,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn page(text: &str) -> Page {
        Page { doc: "D1-nl".into(), lang: "nl".into(), page: 51, text: text.into(), image: PathBuf::new(), has_text: true }
    }

    fn long_text() -> String {
        (0..80)
            .map(|i| format!("Zin nummer {i} gaat over AI-geletterdheid en de verplichtingen van aanbieders."))
            .collect::<Vec<_>>()
            .join(" ")
            + "\n\nEen tweede alinea met één, twee, drie accenten: é ë ï ö."
    }

    #[test]
    fn chunks_respect_the_limit() {
        let chunks = chunk_page(&page(&long_text()), 100, 15);
        assert!(chunks.len() > 2);
        for c in &chunks {
            assert!(char_len(&c.text) <= tokens_to_chars(100), "chunk too long: {}", char_len(&c.text));
        }
    }

    #[test]
    fn consecutive_chunks_overlap() {
        let chunks = chunk_page(&page(&long_text()), 100, 15);
        for pair in chunks.windows(2) {
            let start: String = pair[1].text.chars().take(20).collect();
            assert!(pair[0].text.contains(&start), "no overlap between chunks");
        }
    }

    #[test]
    fn metadata_ids_and_accents() {
        let a = chunk_page(&page(&long_text()), 100, 15);
        let b = chunk_page(&page(&long_text()), 100, 15);
        assert_eq!(a, b, "same input, same chunks and ids");
        assert!(a.iter().all(|c| c.doc == "D1-nl" && c.lang == "nl" && c.page == 51));
        let ids: std::collections::HashSet<u64> = a.iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), a.len(), "ids are unique");
        assert!(ids.iter().all(|&id| id <= i64::MAX as u64), "ids fit in Qdrant's signed payload numbers");
        assert!(a.last().unwrap().text.contains("é ë ï ö"));
    }

    #[test]
    fn empty_page_gives_no_chunks() {
        assert!(chunk_page(&page("  \n\n "), 100, 15).is_empty());
    }
}
```

Note the `>> 1` in `chunk_id`: without it, about half of the ids are bigger than 2^63. Qdrant stores payload numbers as *signed* 64-bit, so those came back as floats and reading the payload failed. That bug really happened while building this guide.

### 2.2 Embeddings (`embed.rs`)

**`crates/ragkit/src/embed.rs`**:

```rust
//! Step 2.2: dense embeddings from bge-m3 (llama.cpp server, OpenAI-style /v1/embeddings).

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::time::Duration;

/// bge-m3 vectors have 1024 numbers.
pub const DIM: usize = 1024;

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

/// Scale a vector to length 1, so cosine similarity is just the dot product.
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    url: String,
}

impl Embedder {
    pub fn new(base_url: &str) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(300)).build()?;
        Ok(Embedder { http, url: format!("{}/v1/embeddings", base_url.trim_end_matches('/')) })
    }

    /// One request: one vector per text, in the same order.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let resp = self
            .http
            .post(&self.url)
            .json(&serde_json::json!({ "input": texts }))
            .send()
            .await
            .with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        if !status.is_success() {
            bail!("embedder returned {status}: {}", resp.text().await.unwrap_or_default());
        }
        let mut items = resp.json::<EmbeddingResponse>().await.context("parsing embeddings")?.data;
        ensure!(items.len() == texts.len(), "asked {} embeddings, got {}", texts.len(), items.len());
        items.sort_by_key(|i| i.index);
        items
            .into_iter()
            .map(|mut item| {
                ensure!(item.embedding.len() == DIM, "expected {DIM} numbers, got {}", item.embedding.len());
                normalize(&mut item.embedding);
                Ok(item.embedding)
            })
            .collect()
    }

    /// Many texts, in batches. The CPU server is slow, so we log progress.
    pub async fn embed_all(&self, texts: &[String], batch: usize) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for (i, part) in texts.chunks(batch).enumerate() {
            out.extend(self.embed(part).await.with_context(|| format!("embedding batch {i}"))?);
            tracing::info!(done = out.len(), total = texts.len(), "embedded");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn vec_of(x: f32) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[0] = x;
        v[1] = x;
        v
    }

    #[tokio::test]
    async fn embeds_in_order_and_normalizes() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"index": 1, "embedding": vec_of(3.0)}, {"index": 0, "embedding": vec_of(1.0)}]
            })))
            .mount(&server)
            .await;
        let e = Embedder::new(&server.uri()).unwrap();
        let v = e.embed(&["a".into(), "b".into()]).await.unwrap();
        assert_eq!(v.len(), 2);
        let norm: f32 = v[0].iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn wrong_count_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"index": 0, "embedding": vec_of(1.0)}]
            })))
            .mount(&server)
            .await;
        let e = Embedder::new(&server.uri()).unwrap();
        assert!(e.embed(&["a".into(), "b".into()]).await.is_err());
    }
}
```

### 2.3 BM25 keyword vectors (`bm25.rs`)

A sparse vector is a list of `(term number, weight)` pairs, one per word in the text. BM25 gives a word more weight when it appears more often, but with diminishing returns (`k1`), and less weight in long chunks (`b`). How rare a word is across all chunks (the IDF) is added by Qdrant.

**`crates/ragkit/src/bm25.rs`**:

```rust
//! Step 2.3: BM25 keyword vectors (sparse vectors). We compute the term-frequency part; Qdrant adds the IDF part
//! (how rare a word is in the whole collection) because the collection is created with `Modifier::Idf`.

use crate::chunk::fnv1a;
use std::collections::BTreeMap;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

const STOPWORDS: &[&str] = &[
    // English
    "the", "of", "and", "to", "in", "a", "an", "is", "are", "was", "be", "for", "on", "by", "with", "as", "at", "or",
    "that", "this", "it", "its", "from", "which", "shall", "what", "who", "how", "does",
    // Dutch
    "de", "het", "een", "en", "van", "in", "op", "te", "dat", "die", "voor", "met", "is", "zijn", "aan", "of", "door",
    "bij", "als", "om", "wat", "wie", "hoe", "worden", "wordt",
];

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseVec {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

fn is_number_like(w: &str) -> bool {
    w.chars().any(|c| c.is_ascii_digit()) && w.chars().all(|c| c.is_ascii_digit() || ".,/%".contains(c))
}

/// Lowercase words. Numbers such as "30.7", "12,500" and "2024/1689" stay one token.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        let w = raw.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if w.is_empty() {
            continue;
        }
        if is_number_like(&w) {
            out.push(w);
            continue;
        }
        for part in w.split(|c: char| !c.is_alphanumeric()) {
            if part.chars().count() >= 2 && !STOPWORDS.contains(&part) {
                out.push(part.to_string());
            }
        }
    }
    out
}

/// Token -> dimension number. Rare collisions are harmless.
pub fn term_id(token: &str) -> u32 {
    let h = fnv1a(token.as_bytes());
    (h ^ (h >> 32)) as u32
}

fn to_sparse(weights: BTreeMap<u32, f32>) -> SparseVec {
    let (indices, values) = weights.into_iter().unzip();
    SparseVec { indices, values }
}

/// A chunk's vector: BM25 term-frequency weight per term. `avg_len` = average token count over all chunks.
pub fn sparse_doc(text: &str, avg_len: f32) -> SparseVec {
    let tokens = tokenize(text);
    let len = tokens.len() as f32;
    let mut tf: BTreeMap<u32, f32> = BTreeMap::new();
    for t in &tokens {
        *tf.entry(term_id(t)).or_default() += 1.0;
    }
    let norm = K1 * (1.0 - B + B * len / avg_len.max(1.0));
    to_sparse(tf.into_iter().map(|(id, f)| (id, f * (K1 + 1.0) / (f + norm))).collect())
}

/// A question's vector: weight 1 per distinct term (Qdrant multiplies by IDF).
pub fn sparse_query(text: &str) -> SparseVec {
    to_sparse(tokenize(text).iter().map(|t| (term_id(t), 1.0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight(v: &SparseVec, token: &str) -> f32 {
        let id = term_id(token);
        v.indices.iter().position(|&i| i == id).map(|p| v.values[p]).unwrap_or(0.0)
    }

    #[test]
    fn tokenize_keeps_numbers_and_drops_stopwords() {
        let t = tokenize("The FTTP coverage of Belgium is 30.7% (Regulation 2024/1689), EUR 12,500.");
        assert!(t.contains(&"30.7%".to_string()) || t.contains(&"30.7".to_string()), "{t:?}");
        assert!(t.contains(&"2024/1689".to_string()));
        assert!(t.contains(&"12,500".to_string()));
        assert!(t.contains(&"belgium".to_string()));
        assert!(!t.contains(&"the".to_string()));
    }

    #[test]
    fn case_does_not_matter() {
        assert_eq!(term_id(&tokenize("Belgium")[0]), term_id(&tokenize("belgium")[0]));
    }

    #[test]
    fn repetition_saturates() {
        let one = sparse_doc("fibre network", 10.0);
        let three = sparse_doc("fibre fibre fibre network", 10.0);
        let ten = sparse_doc(&"fibre ".repeat(10), 10.0);
        assert!(weight(&three, "fibre") > weight(&one, "fibre"));
        assert!(weight(&ten, "fibre") < K1 + 1.0, "BM25 weight never exceeds k1 + 1");
    }

    #[test]
    fn indices_are_sorted_and_unique() {
        let v = sparse_doc("a b c fibre fibre network network", 5.0);
        assert!(v.indices.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(v.indices.len(), v.values.len());
    }
}
```

### 2.4 Qdrant (`store.rs`)

**`crates/ragkit/src/store.rs`**:

```rust
//! Step 2.4: Qdrant. One collection "chunks" with two vectors per point:
//! "dense" (bge-m3, meaning) and "bm25" (sparse keywords; Qdrant applies IDF).

use crate::bm25::SparseVec;
use crate::chunk::Chunk;
use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Modifier, NamedVectors, PointStruct, QueryPointsBuilder, ScoredPoint,
    SparseVectorParamsBuilder, SparseVectorsConfigBuilder, UpsertPointsBuilder, Vector, VectorParamsBuilder,
    VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};

pub const CHUNKS: &str = "chunks";
const DENSE: &str = "dense";
const SPARSE: &str = "bm25";

/// A search result: the chunk and its score (what the score means depends on who produced it).
#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk: Chunk,
    pub score: f32,
}

pub struct Store {
    client: Qdrant,
}

impl Store {
    pub fn connect(url: &str) -> Result<Self> {
        Ok(Store { client: Qdrant::from_url(url).build().context("connecting to Qdrant")? })
    }

    /// Delete and create the "chunks" collection, so re-indexing starts clean.
    pub async fn recreate_chunks(&self, dim: u64) -> Result<()> {
        if self.client.collection_exists(CHUNKS).await? {
            self.client.delete_collection(CHUNKS).await?;
        }
        let mut dense = VectorsConfigBuilder::default();
        dense.add_named_vector_params(DENSE, VectorParamsBuilder::new(dim, Distance::Cosine));
        let mut sparse = SparseVectorsConfigBuilder::default();
        sparse.add_named_vector_params(SPARSE, SparseVectorParamsBuilder::default().modifier(Modifier::Idf));
        self.client
            .create_collection(CreateCollectionBuilder::new(CHUNKS).vectors_config(dense).sparse_vectors_config(sparse))
            .await?;
        Ok(())
    }

    pub async fn upsert_chunks(&self, chunks: &[Chunk], dense: &[Vec<f32>], sparse: &[SparseVec]) -> Result<()> {
        let mut points = Vec::with_capacity(chunks.len());
        for ((chunk, d), s) in chunks.iter().zip(dense).zip(sparse) {
            let vectors = NamedVectors::default()
                .add_vector(DENSE, Vector::new_dense(d.clone()))
                .add_vector(SPARSE, Vector::new_sparse(s.indices.clone(), s.values.clone()));
            let payload = Payload::try_from(serde_json::to_value(chunk)?)?;
            points.push(PointStruct::new(chunk.id, vectors, payload));
        }
        for batch in points.chunks(64) {
            self.client.upsert_points(UpsertPointsBuilder::new(CHUNKS, batch.to_vec()).wait(true)).await?;
        }
        Ok(())
    }

    pub async fn search_dense(&self, query: &[f32], k: u64) -> Result<Vec<Hit>> {
        let req = QueryPointsBuilder::new(CHUNKS).query(query.to_vec()).using(DENSE).limit(k).with_payload(true);
        to_hits(self.client.query(req).await?.result)
    }

    pub async fn search_sparse(&self, query: &SparseVec, k: u64) -> Result<Vec<Hit>> {
        if query.indices.is_empty() {
            return Ok(Vec::new()); // a question with only stopwords
        }
        let pairs: Vec<(u32, f32)> = query.indices.iter().copied().zip(query.values.iter().copied()).collect();
        let req = QueryPointsBuilder::new(CHUNKS).query(pairs).using(SPARSE).limit(k).with_payload(true);
        to_hits(self.client.query(req).await?.result)
    }
}

fn to_hits(points: Vec<ScoredPoint>) -> Result<Vec<Hit>> {
    points
        .into_iter()
        .map(|p| Ok(Hit { score: p.score, chunk: Payload::from(p.payload).deserialize()? }))
        .collect()
}
```

### 2.5 Fusion (`fusion.rs`)

**`crates/ragkit/src/fusion.rs`**:

```rust
//! Step 2.5: Reciprocal Rank Fusion. Dense and sparse scores are on different scales, so we only use positions:
//! score(d) = sum over the lists of 1 / (k + rank), with rank starting at 1.

use crate::store::Hit;
use std::collections::HashMap;

pub const RRF_K: f32 = 60.0;

pub fn rrf(lists: &[Vec<Hit>], k: f32) -> Vec<Hit> {
    let mut fused: HashMap<u64, Hit> = HashMap::new();
    for list in lists {
        for (i, hit) in list.iter().enumerate() {
            let add = 1.0 / (k + i as f32 + 1.0);
            fused
                .entry(hit.chunk.id)
                .and_modify(|h| h.score += add)
                .or_insert_with(|| Hit { chunk: hit.chunk.clone(), score: add });
        }
    }
    let mut out: Vec<Hit> = fused.into_values().collect();
    // Highest score first; ties broken by id so the order is always the same.
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk.id.cmp(&b.chunk.id)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    fn hit(id: u64) -> Hit {
        Hit { chunk: Chunk { id, doc: "D".into(), lang: "en".into(), page: 1, text: String::new() }, score: 0.0 }
    }

    #[test]
    fn found_by_both_beats_found_by_one() {
        let dense = vec![hit(1), hit(2), hit(3)];
        let sparse = vec![hit(4), hit(2), hit(5)];
        let out = rrf(&[dense, sparse], RRF_K);
        assert_eq!(out[0].chunk.id, 2, "2nd in both lists wins over 1st in one");
        assert_eq!(out.len(), 5, "items in only one list are kept");
        assert!(out.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn ties_are_deterministic() {
        let out = rrf(&[vec![hit(9)], vec![hit(3)]], RRF_K);
        assert_eq!(out.iter().map(|h| h.chunk.id).collect::<Vec<_>>(), vec![3, 9]);
    }
}
```

### 2.6 Reranker (`rerank.rs`)

**`crates/ragkit/src/rerank.rs`**:

```rust
//! Step 2.6: a cross-encoder reranker (bge-reranker-v2-m3) reads question and chunk TOGETHER: much more accurate
//! than comparing two separate vectors, but too slow for the whole collection, so only for the top candidates.

use crate::store::Hit;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[derive(Clone)]
pub struct Reranker {
    http: reqwest::Client,
    url: String,
}

impl Reranker {
    pub fn new(base_url: &str) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(300)).build()?;
        Ok(Reranker { http, url: format!("{}/v1/rerank", base_url.trim_end_matches('/')) })
    }

    /// Re-score `hits` against `query` and keep the best `top_n`.
    pub async fn rerank(&self, query: &str, hits: Vec<Hit>, top_n: usize) -> Result<Vec<Hit>> {
        if hits.is_empty() {
            return Ok(hits);
        }
        let documents: Vec<&str> = hits.iter().map(|h| h.chunk.text.as_str()).collect();
        let resp = self
            .http
            .post(&self.url)
            .json(&serde_json::json!({ "query": query, "documents": documents, "top_n": top_n }))
            .send()
            .await
            .with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        if !status.is_success() {
            bail!("reranker returned {status}: {}", resp.text().await.unwrap_or_default());
        }
        let results = resp.json::<RerankResponse>().await.context("parsing rerank response")?.results;

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            ensure!(r.index < hits.len(), "reranker returned index {} for {} documents", r.index, hits.len());
            out.push(Hit { chunk: hits[r.index].chunk.clone(), score: r.relevance_score });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(top_n);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn hit(id: u64) -> Hit {
        Hit { chunk: Chunk { id, doc: "D".into(), lang: "en".into(), page: 1, text: format!("text {id}") }, score: 0.0 }
    }

    #[tokio::test]
    async fn maps_back_by_index() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{"index": 2, "relevance_score": 9.0}, {"index": 0, "relevance_score": -3.0}]
            })))
            .mount(&server)
            .await;
        let r = Reranker::new(&server.uri()).unwrap();
        let out = r.rerank("q", vec![hit(10), hit(11), hit(12)], 2).await.unwrap();
        assert_eq!(out.iter().map(|h| h.chunk.id).collect::<Vec<_>>(), vec![12, 10]);
        assert_eq!(out[0].score, 9.0);
    }

    #[tokio::test]
    async fn bad_index_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{"index": 7, "relevance_score": 1.0}]
            })))
            .mount(&server)
            .await;
        let r = Reranker::new(&server.uri()).unwrap();
        assert!(r.rerank("q", vec![hit(1)], 1).await.is_err());
    }
}
```

### 2.7 The pipeline (`vector.rs`) and the final CLI for this chapter

**`crates/ragkit/src/vector.rs`**:

```rust
//! Step 2.7: the vector RAG pipeline.
//! index: pages -> chunks -> dense + sparse vectors -> Qdrant
//! ask:   question -> dense top-20 + sparse top-20 -> RRF -> rerank to top-5 -> LLM with only those 5 chunks

use crate::bm25::{self, SparseVec};
use crate::chunk::{self, Chunk};
use crate::corpus::Page;
use crate::embed::{self, Embedder};
use crate::fusion::{RRF_K, rrf};
use crate::llm::{ChatAnswer, ChatOptions, Llm, Message};
use crate::longctx::SYSTEM_RULES;
use crate::rerank::Reranker;
use crate::store::{Hit, Store};
use anyhow::{Context, Result};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retrieval {
    Dense,
    Sparse,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct VectorSettings {
    pub retrieval: Retrieval,
    /// How many candidates each search returns, and how many go to the reranker.
    pub candidates: u64,
    /// How many chunks the LLM finally sees.
    pub top_k: usize,
    pub rerank: bool,
}

impl Default for VectorSettings {
    fn default() -> Self {
        VectorSettings { retrieval: Retrieval::Hybrid, candidates: 20, top_k: 5, rerank: true }
    }
}

/// One source the answer was built from, for printing and for the eval.
#[derive(Debug, Clone)]
pub struct Source {
    pub cite: String,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub retrieve: Duration,
    pub rerank: Duration,
    pub generate: Duration,
}

#[derive(Debug, Clone)]
pub struct RagAnswer {
    pub answer: ChatAnswer,
    pub sources: Vec<Source>,
    pub timings: Timings,
}

#[derive(Debug)]
pub struct IndexStats {
    pub pages: usize,
    pub chunks: usize,
    pub elapsed: Duration,
}

/// The chunks as tagged blocks for the prompt.
pub fn render_chunks(hits: &[Hit]) -> String {
    hits.iter()
        .map(|h| format!("<chunk doc=\"{}\" page=\"{}\">\n{}\n</chunk>\n", h.chunk.doc, h.chunk.page, h.chunk.text.trim()))
        .collect()
}

pub struct VectorRag {
    pub embedder: Embedder,
    pub reranker: Reranker,
    pub store: Store,
    pub llm: Llm,
    pub settings: VectorSettings,
}

impl VectorRag {
    pub async fn index(&self, pages: &[Page]) -> Result<IndexStats> {
        let started = Instant::now();
        let chunks: Vec<Chunk> =
            pages.iter().flat_map(|p| chunk::chunk_page(p, chunk::MAX_TOKENS, chunk::OVERLAP_TOKENS)).collect();
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

        let total_tokens: usize = texts.iter().map(|t| bm25::tokenize(t).len()).sum();
        let avg_len = total_tokens as f32 / texts.len().max(1) as f32;
        let sparse: Vec<SparseVec> = texts.iter().map(|t| bm25::sparse_doc(t, avg_len)).collect();
        let dense = self.embedder.embed_all(&texts, 16).await?;

        self.store.recreate_chunks(embed::DIM as u64).await?;
        self.store.upsert_chunks(&chunks, &dense, &sparse).await?;
        Ok(IndexStats { pages: pages.len(), chunks: chunks.len(), elapsed: started.elapsed() })
    }

    async fn search_dense(&self, question: &str) -> Result<Vec<Hit>> {
        let v = self.embedder.embed(&[question.to_string()]).await?.pop().context("no embedding")?;
        self.store.search_dense(&v, self.settings.candidates).await
    }

    async fn search_sparse(&self, question: &str) -> Result<Vec<Hit>> {
        self.store.search_sparse(&bm25::sparse_query(question), self.settings.candidates).await
    }

    /// Search only: the candidates for a question (no reranking, no LLM).
    pub async fn retrieve(&self, question: &str, retrieval: Retrieval) -> Result<Vec<Hit>> {
        Ok(match retrieval {
            Retrieval::Dense => self.search_dense(question).await?,
            Retrieval::Sparse => self.search_sparse(question).await?,
            Retrieval::Hybrid => {
                let (dense, sparse) = tokio::try_join!(self.search_dense(question), self.search_sparse(question))?;
                let mut fused = rrf(&[dense, sparse], RRF_K);
                fused.truncate(self.settings.candidates as usize);
                fused
            }
        })
    }

    pub async fn ask(&self, question: &str) -> Result<RagAnswer> {
        let mut timings = Timings::default();

        let t = Instant::now();
        let candidates = self.retrieve(question, self.settings.retrieval).await?;
        timings.retrieve = t.elapsed();

        let t = Instant::now();
        let top = if self.settings.rerank {
            self.reranker.rerank(question, candidates, self.settings.top_k).await?
        } else {
            candidates.into_iter().take(self.settings.top_k).collect()
        };
        timings.rerank = t.elapsed();

        let t = Instant::now();
        let messages = [
            Message::system(SYSTEM_RULES),
            Message::user(format!("Documents:\n{}\nQuestion: {question}", render_chunks(&top))),
        ];
        let answer = self.llm.chat(&messages, &ChatOptions::default()).await?;
        timings.generate = t.elapsed();

        let sources = top.iter().map(|h| Source { cite: h.chunk.cite(), score: h.score }).collect();
        Ok(RagAnswer { answer, sources, timings })
    }
}
```

**`crates/ragkit/src/lib.rs`**:

```rust
//! ragkit: all the RAG logic. The `rag` binary only parses arguments and calls into this crate.

pub mod config;
pub mod corpus;
pub mod prepare;

pub mod llm;
pub mod longctx;

pub mod bm25;
pub mod chunk;
pub mod embed;
pub mod fusion;
pub mod rerank;
pub mod store;
pub mod vector;
```

**`crates/rag/src/main.rs`** (replace the whole file):

```rust
//! `rag`: the command line. Only argument parsing and printing; the logic lives in ragkit.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use ragkit::config::Config;
use ragkit::vector::{RagAnswer, Retrieval, VectorRag, VectorSettings};
use ragkit::{corpus, embed, llm, longctx, prepare, rerank, store};
use std::io::Write;

#[derive(Parser)]
#[command(name = "rag", about = "RAG three ways: long-context, vector, multimodal")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// PDFs -> one .txt and one .png per page, plus the fake scan and the manifest
    Prepare,
    /// Build the search index for a mode
    Index {
        #[arg(long, value_enum, default_value = "vector")]
        mode: Mode,
        #[arg(long, default_value = "small")]
        subset: String,
    },
    /// Ask a question. Without a question: read questions from stdin, one per line
    Ask {
        #[arg(long, value_enum, default_value = "lc")]
        mode: Mode,
        #[arg(long, default_value = "small")]
        subset: String,
        /// vector mode: which search to use
        #[arg(long, value_enum, default_value = "hybrid")]
        retrieval: RetrievalArg,
        /// vector mode: skip the reranker
        #[arg(long)]
        no_rerank: bool,
        question: Option<String>,
    },
    /// vector mode: the top pages of dense, sparse, hybrid and hybrid+rerank side by side (no LLM)
    Search { question: String },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Mode {
    /// No documents at all (shows why RAG is needed)
    Plain,
    /// Long-context: the whole subset in the prompt
    Lc,
    /// Vector RAG: chunks, hybrid search, rerank
    Vector,
}

#[derive(Clone, Copy, ValueEnum)]
enum RetrievalArg {
    Dense,
    Sparse,
    Hybrid,
}

impl From<RetrievalArg> for Retrieval {
    fn from(r: RetrievalArg) -> Self {
        match r {
            RetrievalArg::Dense => Retrieval::Dense,
            RetrievalArg::Sparse => Retrieval::Sparse,
            RetrievalArg::Hybrid => Retrieval::Hybrid,
        }
    }
}

fn vector_rag(cfg: &Config, settings: VectorSettings) -> Result<VectorRag> {
    Ok(VectorRag {
        embedder: embed::Embedder::new(&cfg.embed_url)?,
        reranker: rerank::Reranker::new(&cfg.rerank_url)?,
        store: store::Store::connect(&cfg.qdrant_url)?,
        llm: llm::Llm::new(&cfg.llm_url, &cfg.llm_model)?,
        settings,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cfg = Config::from_env();

    match Cli::parse().command {
        Command::Prepare => {
            let pdfium = prepare::bind_pdfium(&cfg.pdfium_dir)?;
            let pages = prepare::prepare_all(&pdfium, &cfg.corpus_dir)?;
            println!("prepared {} pages into {}/pages", pages.len(), cfg.corpus_dir.display());
        }

        Command::Index { mode, subset } => {
            let pages = corpus::load_pages(&cfg.corpus_dir, &subset)?;
            match mode {
                Mode::Vector => {
                    let stats = vector_rag(&cfg, VectorSettings::default())?.index(&pages).await?;
                    println!("indexed {} pages as {} chunks in {:.1} s", stats.pages, stats.chunks, stats.elapsed.as_secs_f64());
                }
                Mode::Plain | Mode::Lc => println!("this mode needs no index"),
            }
        }

        Command::Ask { mode, subset, retrieval, no_rerank, question } => {
            let interactive = question.is_none();
            let mut one = question;
            match mode {
                Mode::Plain => {
                    let llm = llm::Llm::new(&cfg.llm_url, &cfg.llm_model)?;
                    while let Some(q) = next_question(&mut one, interactive) {
                        let a = llm.chat(&[llm::Message::user(q)], &llm::ChatOptions::default()).await?;
                        println!("{}\n  [{:.1} s]", a.text, a.elapsed.as_secs_f64());
                    }
                }
                Mode::Lc => {
                    let llm = llm::Llm::new(&cfg.llm_url, &cfg.llm_model)?;
                    let pages = corpus::load_pages(&cfg.corpus_dir, &subset)?;
                    let lc = longctx::LongContext::new(llm, &pages)?;
                    while let Some(q) = next_question(&mut one, interactive) {
                        let a = lc.ask(&q).await?;
                        println!("{}", a.text);
                        println!(
                            "  [{:.1} s, {} prompt tokens, {} cached, cache {}]",
                            a.elapsed.as_secs_f64(),
                            a.prompt_tokens,
                            a.cached_tokens,
                            if longctx::is_cache_hit(&a) { "hit" } else { "miss" }
                        );
                    }
                }
                Mode::Vector => {
                    let settings = VectorSettings { retrieval: retrieval.into(), rerank: !no_rerank, ..Default::default() };
                    let rag = vector_rag(&cfg, settings)?;
                    while let Some(q) = next_question(&mut one, interactive) {
                        print_rag_answer(&rag.ask(&q).await?);
                    }
                }
            }
        }

        Command::Search { question } => {
            let rag = vector_rag(&cfg, VectorSettings::default())?;
            let pages = |hits: &[store::Hit]| hits.iter().take(5).map(|h| h.chunk.cite()).collect::<Vec<_>>();
            let dense = rag.retrieve(&question, Retrieval::Dense).await?;
            let sparse = rag.retrieve(&question, Retrieval::Sparse).await?;
            let hybrid = rag.retrieve(&question, Retrieval::Hybrid).await?;
            println!("dense:          {:?}", pages(&dense));
            println!("sparse:         {:?}", pages(&sparse));
            println!("hybrid:         {:?}", pages(&hybrid));
            let reranked = rag.reranker.rerank(&question, hybrid, 5).await?;
            println!("hybrid+rerank:  {:?}", pages(&reranked));
        }
    }
    Ok(())
}

fn print_rag_answer(r: &RagAnswer) {
    println!("{}", r.answer.text);
    let sources: Vec<String> = r.sources.iter().map(|s| format!("{} ({:.2})", s.cite, s.score)).collect();
    println!("  sources: {}", sources.join(", "));
    println!(
        "  [retrieve {:.2} s, rerank {:.1} s, generate {:.1} s, {} prompt tokens]",
        r.timings.retrieve.as_secs_f64(),
        r.timings.rerank.as_secs_f64(),
        r.timings.generate.as_secs_f64(),
        r.answer.prompt_tokens
    );
}

/// The next question: the one from the command line (once), or the next line typed on stdin.
fn next_question(one: &mut Option<String>, interactive: bool) -> Option<String> {
    if !interactive {
        return one.take();
    }
    loop {
        print!("question> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).ok()? == 0 {
            return None; // Ctrl-D
        }
        if !line.trim().is_empty() {
            return Some(line.trim().to_string());
        }
    }
}
```

**Check:**

```bash
cargo test -p ragkit                       # 26 passed
cargo build --release -p rag
./target/release/rag index --subset small  # "indexed 25 pages as 65 chunks in ~16 s"
./target/release/rag search "Wat moeten aanbieders doen om AI-geletterdheid te waarborgen?"
```

What this PC printed for the search:

```
dense:          ["D1-en p.51", "D1-nl p.51", "D1-en p.46", "D1-en p.1", "D5-acme p.1"]
sparse:         ["D1-nl p.51", "D1-nl p.115", "D1-nl p.115", "D1-nl p.52", "D5-acme p.1"]
hybrid:         ["D1-nl p.51", "D1-en p.51", "D5-acme p.1", "D1-en p.46", "D1-nl p.115"]
hybrid+rerank:  ["D1-nl p.51", "D1-en p.51", "D1-en p.1", "D1-en p.1", "D1-en p.46"]
```

This is the lesson of the chapter:

- **Dense** puts the *English* Article 4 first. It matches meaning across languages, which is great for an English question about a Dutch text, but here the user wants Dutch.
- **Sparse** puts the Dutch page first: it matches the exact words.
- **Hybrid** keeps the best of both, and the reranker agrees.

Now full answers:

```bash
./target/release/rag ask --mode vector
```

Ask the same three questions as in chapter 01. On this PC: the right Acme answer citing `[D5-acme p.1]`; the Dutch answer now cites the right page, `[D1-nl p.51]`; and the FTTP question gives `NOT FOUND` again. Each answer prints its sources and timings: retrieval takes ~0.03 s, the reranker ~5.5 s on the CPU (that's the cost of accuracy), and generation 1–4 s with only ~1,600–2,100 prompt tokens instead of 17,500.

Try the experiments and write the results in `PROGRESS.md`:

```bash
./target/release/rag ask --mode vector --retrieval dense  "Which threshold of EUR 12,500 applies to AI approvals?"
./target/release/rag ask --mode vector --retrieval sparse "Which threshold of EUR 12,500 applies to AI approvals?"
./target/release/rag ask --mode vector --no-rerank        "Wat is de maximale boete voor verboden AI-praktijken?"
./target/release/rag index --subset full    # vector RAG doesn't care about corpus size
```

**Chapter 02 is done.** Commit.

---

## Chapter 03: Multimodal RAG (outline)

The code for this chapter will be added once it has been verified on this PC, the same way as chapters 00–02.

**Idea:** text extraction loses tables, charts and scans. Treat every page as an **image** instead. ColPali turns a page image into ~1,000 small vectors (one per image patch) and a question into one vector per word. **MaxSim** matches each question word with its best patch. The best pages then go, as images, to the vision model, which reads the answer off the page.

| Step | What | Notes |
|---|---|---|
| 3.1 | `textify.rs`: for pages without text (the scan), ask Qwen3-VL to transcribe the image to Markdown and index that text | Tested on this PC: from the KPI-table image the model read **30.7** correctly in 4.4 s (~2,200 tokens per page) |
| 3.2 | `crates/colpali-server`: ColPali in Rust (candle), served over HTTP | Must fit in the ~3.5 GB of VRAM that's left next to the LLM, so it runs quantized |
| 3.3 | `maxsim.rs` | pure function, unit tests |
| 3.4 | `store.rs`: a `pages` collection with multivectors (MaxSim comparator) | |
| 3.5 | `multimodal.rs`: question → ColPali → top pages → images + question → Qwen3-VL | |

## Chapter 04: Compare, package, explain (outline)

| Step | What |
|---|---|
| 4.1 | Grow `golden.jsonl` to 40 questions in 7 categories |
| 4.2 | `rag eval`: every question through every mode; score on `must_contain`; save each result as soon as it's done; summary table |
| 4.3 | `docker/app.Dockerfile` (distroless, non-root), an `app` service, `rag serve` (axum, `/ask`, `/health`) |
| 4.4 | README: the results, and which RAG to use when |
| 4.5 | `docker compose down`; `nvidia-smi` shows the GPU free |

---

## Lessons learned while building this

1. **A 35B model with pinned RAM froze the 31 GB PC** once everything else ran next to it. An 8B Q4 model fully on the GPU leaves the RAM free and is fast enough.
2. **Hybrid models make caching hard.** The first version used Strata's Qwen3.6-35B (a hybrid DeltaNet model). Its cache only worked with a special request shape (the corpus as a tool result, then the question as a second tool result). Qwen3-VL-8B is a plain transformer: normal prefix caching just works.
3. **Qdrant payload numbers are signed 64-bit.** Ids above 2^63 come back as floats (2.1).
4. **EUR-Lex blocks scripted downloads.** The Publications Office serves the same PDF (0.4).
5. **Debug builds are slow for image work.** `rag prepare` took 4 minutes in debug and 7 seconds in release.
6. **Keep the cached part byte-for-byte identical.** Same pages, same order, same text, every time (1.2).
7. **The reranker costs ~5.5 s on the CPU** for 20 candidates. It's worth it for quality, but measure it.
