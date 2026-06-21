# Basic RAG Roadmap

This roadmap tracks the first retrieval-augmented generation (RAG) system in
this repository. The initial goal is a small, understandable baseline. More
advanced retrieval and evaluation techniques can be added after the complete
pipeline works end to end.

## 1. Load documents

Load the tutorial PDFs with `PyPDFLoader`. Each PDF page becomes a LangChain
`Document`.

- Preserve source, page number, title, and arXiv metadata.
- Confirm that all expected files and pages are loaded.
- Status: implemented in `notebooks/rag_basics.ipynb`.

## 2. Split documents

Split pages into smaller retrieval units with
`RecursiveCharacterTextSplitter`.

- Baseline: `chunk_size=1000`, `chunk_overlap=200`.
- Preserve page metadata on every chunk for later citations.
- Inspect chunk sizes and sample text before embedding.
- Status: implemented in `notebooks/rag_basics.ipynb`.

## 3. Create embeddings

Use an embedding model to convert each chunk into a numerical vector.

- Default to the local
  `sentence-transformers/all-MiniLM-L6-v2` Hugging Face model.
- Allow `OpenAIEmbeddings` with `text-embedding-3-small` as an optional
  provider selected through notebook configuration.
- Avoid regenerating embeddings when the source documents are unchanged.
- Status: provider selection and embedding initialization are implemented in
  `notebooks/rag_basics.ipynb`.

## 4. Build the vector store

Store chunks and embeddings in ChromaDB.

- Persist the index under `data/processed/chroma/`.
- Keep separate indexes for each embedding provider and model so vector
  dimensions are never mixed.
- Reopen an existing collection on later runs.
- Record the embedding model and chunking configuration used to build it.
- Fingerprint the chunk contents and require an explicit rebuild when the
  configuration changes.
- Status: implemented in `notebooks/rag_basics.ipynb`.

## 5. Configure and test retrieval

Create a Chroma retriever using similarity search.

- Start with `k=4`.
- Test retrieval independently from answer generation.
- Print retrieved text, source PDF, page number, and similarity information
  when available.
- Include one page-specific smoke-test query per tutorial paper.
- Status: initial five-paper similarity retrieval test implemented in
  `notebooks/rag_basics.ipynb`.

## 6. Define the RAG prompt

Construct a prompt containing the retrieved context and user question.

- Require answers to use only the supplied context.
- Instruct the model to say when the context is insufficient.
- Require citations using source names and page numbers.

## 7. Generate cited answers

Connect the retriever, prompt, and chat model into a basic RAG chain.

- Return the answer and supporting source documents.
- Keep retrieval results inspectable for debugging.
- Avoid hiding the pipeline behind an agent until the baseline is reliable.

## 8. Test the complete pipeline

Use several types of questions:

- Questions answered by one paper.
- Questions requiring evidence from multiple papers.
- Questions not covered by the collection.
- Questions where similar terminology appears in different papers.

Check that answers are grounded and citations point to relevant pages.

## 9. Add basic evaluation

Create a small regression dataset of questions, expected facts, and expected
sources.

- Evaluate retrieval separately from generation.
- Record chunking, embedding, retrieval, and model settings.
- Re-run the dataset whenever the pipeline changes.

## Later extensions

After the baseline is working, consider metadata filtering, maximal marginal
relevance, hybrid BM25/vector search, reranking, contextual compression,
query rewriting, parent-document retrieval, semantic chunking, LangSmith
evaluation, conversational memory, and LangGraph orchestration.
