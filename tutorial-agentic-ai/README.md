# Agentic AI Tutorial

A growing collection of practical examples for building LLM applications with
LangChain and LangGraph. The repository starts with model API calls and will
expand into retrieval-augmented generation (RAG), tool-using agents, persistent
workflows, evaluation, and deployable applications.

## Technology stack

- **LangChain** for model integrations, prompts, tools, document loading, and
  retrieval pipelines.
- **LangGraph** for stateful agents and controllable multi-step workflows.
- **ChromaDB** for local vector storage and semantic retrieval.
- **OpenAI, Google Gemini, and Groq** model integrations.
- **LangSmith** for tracing and evaluation.
- **Streamlit** for simple interactive applications.
- **Jupyter** for experiments and tutorial notebooks.

Supporting RAG packages include `pypdf`, Beautiful Soup, `lxml`, `tiktoken`,
LangChain text splitters, and `rank-bm25` for keyword or hybrid retrieval.

## Setup

This project uses Python 3.11 and
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Create a local environment file from the template:

```bash
cp .env.example .env
```

Add only the API keys needed for the provider you are using. Never commit the
`.env` file.

Run the starter module:

```bash
uv run python main.py
```

## Download the tutorial papers

The initial RAG examples use five arXiv papers. Download them into the default
raw-data directory with:

```bash
uv run python -m src.download_arxiv_papers
```

The command creates `data/raw/arxiv/` and stores each paper using its arXiv ID,
for example `data/raw/arxiv/2606.20497.pdf`. Existing files are skipped so the
command is safe to run repeatedly.

To choose another destination or replace existing files:

```bash
uv run python -m src.download_arxiv_papers --output-dir path/to/papers
uv run python -m src.download_arxiv_papers --overwrite
```

Start Jupyter:

```bash
uv run jupyter notebook
```

Start a Streamlit application when one is available:

```bash
uv run streamlit run apps/<app_name>/app.py
```

## Current contents

```text
.
├── main.py
├── notebooks/
│   ├── call_api.ipynb
│   └── rag_basics.ipynb
├── src/
│   └── download_arxiv_papers.py
├── tests/
│   └── test_download_arxiv_papers.py
├── pyproject.toml
└── uv.lock
```

## Planned tutorial path

1. Model API calls, messages, prompts, and structured output
2. Chains, tools, and runnable composition
3. LangGraph state, nodes, edges, routing, and persistence
4. RAG ingestion: load, clean, split, embed, and index documents
5. RAG retrieval: similarity search, filtering, reranking, and hybrid search
6. Conversational RAG with memory and citations
7. Tool-using and human-in-the-loop agents
8. Agent evaluation, tracing, testing, and observability
9. Streamlit applications and deployment

As examples are added, reusable code should go in `src/`, exploratory material
in `notebooks/`, user-facing demos in `apps/`, and tests in `tests/`.

## Dependency management

Add a runtime package and update the lockfile:

```bash
uv add <package>
```

Add development tooling:

```bash
uv add --dev <package>
```

Run the current quality checks:

```bash
uv run ruff check .
uv run pytest
```

The committed `uv.lock` keeps tutorial environments reproducible. Run
`uv sync` after pulling dependency changes.

## Data and generated files

Do not commit API keys, local vector databases, downloaded source documents,
model artifacts, or generated application data. Store these under ignored
directories such as `data/`, `.chroma/`, or `artifacts/`.
