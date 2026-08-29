# Shelfmate — Content-Based Book Recommendation System

Search for a book you know, and Shelfmate finds what belongs beside it on the shelf — powered by sentence embeddings and cosine similarity over the [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset.

## How it works

1. **Data prep** — Books are merged with their most frequent tags (from `book_tags.csv` + `tags.csv`) into a single content string: `title + authors + top tags`.
2. **Embeddings** — Each book's content string is encoded into a 384-dimensional vector using `sentence-transformers` (`all-MiniLM-L6-v2`, PyTorch backend). This is feature extraction / transfer learning — no model is trained from scratch.
3. **Recommendation** — Given a book title, its embedding is compared against all others via cosine similarity; the closest matches are returned.
4. **API** — A FastAPI backend exposes `/search` (title lookup/autocomplete) and `/recommend` (similar-books lookup) endpoints.
5. **Frontend** — A single-page HTML/CSS/JS interface (glassmorphic, red theme) calls the API and displays results as a browsable grid.

## Project structure

```
project_root/
├── book/               # raw Goodbooks-10k CSVs (books, tags, book_tags, ratings)
├── model/               # processed data + generated embeddings
│   ├── embeddings.npy
│   └── books_processed.pkl
├── schema/
│   ├── __init__.py
│   └── main.py           # FastAPI app
├── index.html             # frontend
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### 1. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 2. Generate embeddings (if not already done)
Run the data-prep + embedding notebook/script to produce `model/embeddings.npy` and `model/books_processed.pkl`.

### 3. Run the API
From the project root:
```bash
uvicorn schema.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`.

### 4. Open the frontend
Open `index.html` directly in a browser. Make sure `API_BASE` in the script matches your running API URL, and that CORS is enabled in `schema/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /search?query=<text>` | Returns up to 10 books whose title matches the query (for autocomplete). |
| `GET /recommend?book_title=<exact title>&top_n=5` | Returns the `top_n` most similar books to the given title, by content similarity. |

**Note:** Both endpoints only work within the 10,000 books in the Goodbooks-10k dataset — unknown titles return a 404.

## Docker

Build and run:
```bash
docker build -t shelfmate .
docker run -p 8000:8000 shelfmate
```

## Tech stack

- **ML:** sentence-transformers, scikit-learn (cosine similarity), pandas, numpy
- **Backend:** FastAPI, Uvicorn
- **Frontend:** vanilla HTML/CSS/JS
- **Deployment:** Docker

## Limitations & future improvements

- Recommendations are limited to books already in the dataset — no live lookup for unknown titles.
- Purely content-based; doesn't account for user rating patterns (a collaborative-filtering or hybrid layer could improve relevance).
- Could add a vector index (e.g. FAISS) for faster similarity search at larger scale.