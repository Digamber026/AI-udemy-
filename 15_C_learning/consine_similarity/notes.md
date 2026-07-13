# Cosine Similarity — Concept Notes

## What is an Embedding?

Computers can't understand raw text. An **embedding model** converts a sentence into a vector of numbers that captures its semantic meaning.

```
"What is the name of the cat?"  →  [0.12, -0.45, 0.87, ..., 0.31]
```

Similar meanings → similar vectors (pointing in similar directions).

---

## What is Cosine Similarity?

Cosine similarity measures the **angle** between two vectors, not their magnitude. It tells us how similar two pieces of text are, regardless of their length.

```
Cosine Similarity = Dot Product / (|A| × |B|)
```

Where:
- **Dot Product** = sum(Aᵢ × Bᵢ)
- **|A|** = √(A₁² + A₂² + ... + Aₙ²)  → magnitude (length) of vector A
- **|B|** = √(B₁² + B₂² + ... + Bₙ²)  → magnitude (length) of vector B

### Score Range: -1 to 1

| Score | Meaning |
|-------|---------|
| **1**  | Same meaning / same direction |
| **0**  | Unrelated |
| **-1** | Opposite direction (rare in practice) |

---

## Why Can the Score Be Negative or Near-Zero?

Embedding values contain both positive and negative numbers. If two sentences are unrelated, their vectors may point in slightly different directions, giving a dot product close to (or slightly below) zero.

Example:
```
"What is the name of the cat?"  vs  "What is the capital of India?"
Cosine Similarity ≈ -0.0026
```

A small negative number does **not** mean the sentences are opposites — it just means they're **almost unrelated**.

---

## Workflow Overview

```
Sentence → SentenceTransformer → Embedding Vector
                                        │
                     Dot Product   Magnitude(A)   Magnitude(B)
                                        │
                              Cosine Similarity
```

---

## Suggested Folder Structure

```
Learning_/
│
├── main.py
└── venv11/
```

---

## Useful Terminal Commands

| Purpose | Command |
|---|---|
| Create virtual environment | `python3 -m venv venv11` |
| Activate venv (macOS/Linux) | `source venv11/bin/activate` |
| Install sentence-transformers | `pip install sentence-transformers` |
| Run script | `python main.py` or `python3 main.py` |
| Deactivate venv | `deactivate` |

---

## Key Takeaways

- Text must be converted into embeddings before similarity can be computed.
- Cosine similarity compares **direction**, not magnitude — it's scale-invariant.
- Manual implementation builds intuition for what libraries like `sklearn` do under the hood.
- This concept is foundational to **semantic search** and **Retrieval-Augmented Generation (RAG)**.

---

## What to Learn Next

1. Compare one query against multiple documents.
2. Sort documents by cosine similarity score.
3. Return the Top-K most similar documents.
4. Read documents from a text file.
5. Read and chunk a PDF.
6. Store embeddings persistently.
7. Learn a vector database (FAISS, Qdrant).
8. Build a complete RAG application.