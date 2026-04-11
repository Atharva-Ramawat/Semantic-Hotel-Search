# frontend_streamlit/search.py
import numpy as np
import re

def execute_search(query, df, model, index, top_k=12):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    distances, indices = index.search(query_vector, k=len(df))
    results_df = df.iloc[indices[0]].copy()
    
    query_lower = query.lower()
    all_countries = [str(c).strip() for c in df['countyName'].unique() if str(c).strip()]
    detected_countries = [c for c in all_countries if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]
    
    all_cities = [str(c).strip() for c in df['cityName'].unique() if str(c).strip()]
    detected_cities = [c for c in all_cities if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]

    if detected_countries:
        results_df = results_df[results_df['countyName'].astype(str).str.contains('|'.join(detected_countries), case=False, na=False)]
    if detected_cities:
        results_df = results_df[results_df['cityName'].astype(str).str.contains('|'.join(detected_cities), case=False, na=False)]

    top_matches = results_df.head(top_k)
    return top_matches, detected_countries, detected_cities