# Repository Guidelines

## Project Structure & Module Organization

This repository contains hands-on examples for LangChain, LangGraph, and
retrieval-augmented generation (RAG).

- `src/`: reusable Python modules, including data acquisition and future RAG
  components.
- `notebooks/`: tutorial notebooks and exploratory examples.
- `tests/`: pytest tests corresponding to modules in `src/`.
- `data/raw/arxiv/`: downloaded tutorial PDFs; `data/` is generated locally
  and excluded from Git.
- `main.py`: minimal project entry point.
- `pyproject.toml` and `uv.lock`: dependencies and reproducible environment.

Move reusable notebook logic into `src/`. Keep generated indexes under
`data/processed/`, `.chroma/`, or `artifacts/`.

## Build, Test, and Development Commands

Use Python 3.11 and `uv`:

```bash
uv sync
uv run python main.py
uv run jupyter notebook
```

Download the tutorial papers with:

```bash
uv run python -m src.download_arxiv_papers
```

Run validation before committing:

```bash
uv run ruff check .
uv run pytest
uv run python -m compileall src
```

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>`
for development tools. Commit the updated `uv.lock`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use type hints, focused functions,
deterministic behavior, and `pathlib.Path` instead of hard-coded path strings.
Use `snake_case` for modules, functions, variables, notebook names, and CLI
options; use `PascalCase` for classes and `UPPER_CASE` for constants. Ruff is
the repository linter.

Notebooks should be readable tutorials: introduce each stage with Markdown,
keep cells small, and save useful execution outputs. Do not embed API keys or
machine-specific absolute paths.

## Testing Guidelines

Tests use pytest and belong in `tests/`, named `test_<module>.py`. Test
functions should describe behavior, for example
`test_download_pdf_rejects_non_pdf_response`. Mock network and model calls
where practical; tests should not require credentials or download large files.
No coverage threshold is currently enforced, but new nontrivial `src/` logic
should include focused tests.

## Commit & Pull Request Guidelines

History uses short, imperative subjects such as `Add Home Credit model
experiments`. Keep commits focused and use subjects like `Add basic RAG
retriever`. Pull requests should summarize the tutorial stage changed, list
validation commands, identify new dependencies or environment variables, and
include screenshots only when notebook or Streamlit output needs visual review.

## Security & Generated Data

Copy `.env.example` to `.env` and keep credentials local. Never commit `.env`,
downloaded PDFs, Chroma indexes, generated outputs, or model artifacts.
