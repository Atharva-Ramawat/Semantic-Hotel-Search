import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os

# Safely locate the CSV file one directory above this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', 'hotels.csv')

print("🧠 Loading AI Engine and Data Core. Please wait...")
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding='latin1')

# Initialize Data
df = df.sample(n=10000, random_state=42)
df.columns = df.columns.str.strip()
df = df.fillna('')
df['Search_Text'] = df['HotelName'].astype(str) + " " + \
                    df['Description'].astype(str) + " " + \
                    df['HotelFacilities'].astype(str) + " " + \
                    df['cityName'].astype(str) + " " + \
                    df['countyName'].astype(str)

# Initialize AI & Vector DB
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Search_Text'].tolist(), convert_to_tensor=False)
embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("✅ AI Engine Ready!")

def perform_search(query: str, top_k: int):
    """Core business logic for semantic search and regex filtering."""
    # Convert incoming query to vector
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    # Search Vector Database
    distances, indices = index.search(query_vector, len(df))
    results_df = df.iloc[indices[0]].copy()
    
    # Smart Location Filters
    query_lower = query.lower()
    all_countries = [str(c).strip() for c in df['countyName'].unique() if str(c).strip()]
    detected_countries = [c for c in all_countries if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]
    all_cities = [str(c).strip() for c in df['cityName'].unique() if str(c).strip()]
    detected_cities = [c for c in all_cities if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]

    if detected_countries:
        results_df = results_df[results_df['countyName'].astype(str).str.contains('|'.join(detected_countries), case=False, na=False)]
    if detected_cities:
        results_df = results_df[results_df['cityName'].astype(str).str.contains('|'.join(detected_cities), case=False, na=False)]

    # Get top matches and format
    top_matches = results_df.head(top_k)
    top_matches['Description'] = top_matches['Description'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' ').strip()[:300] + "...")
    
    return {
        "filters_applied": {"countries": detected_countries, "cities": detected_cities},
        "matches": top_matches[['HotelName', 'cityName', 'countyName', 'HotelRating', 'HotelFacilities', 'Description']].to_dict(orient="records")
    }