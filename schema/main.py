# main.py
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent  # schema se ek level upar = project root
MODEL_DIR = BASE_DIR / "model"

embeddings = np.load(MODEL_DIR / "embeddings.npy")
books = pd.read_pickle(MODEL_DIR / "books_processed.pkl")

@app.get("/recommend")
def recommend(book_title: str, top_n: int = 5):
    idx = books[books['title'].str.lower() == book_title.lower()].index
    
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail=f"'{book_title}' not found")
    
    idx = idx[0]
    book_embedding = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(book_embedding, embeddings)[0]
    
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    
    results = books.iloc[similar_indices][['title', 'authors', 'average_rating']].copy()
    results['similarity_score'] = similarities[similar_indices].tolist()
    
    return results.to_dict(orient='records')

@app.get("/search")
def search(query: str):
    matches = books[books['title'].str.contains(query, case=False, na=False)]
    return matches[['title', 'authors']].head(10).to_dict(orient='records')