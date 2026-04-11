import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import json

# --- UI Configuration (Premium Dark Mode) ---
st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    /* Premium Ocean Slate Theme - High Info Density Grid */
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    h1, h2, h3 { color: #38BDF8 !important; font-weight: 600; letter-spacing: -0.5px; }
    
    .hotel-card { 
        background-color: #1E293B; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #334155;
        border-left: 5px solid #38BDF8; 
        margin-bottom: 20px; 
        height: 340px; /* Adjusted for better spacing */
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hotel-card:hover { 
        transform: translateY(-4px); 
        border-left: 5px solid #7DD3FC;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .hotel-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
        gap: 10px;
    }
    
    .hotel-name { 
        font-size: 18px; 
        font-weight: 700; 
        color: #F8FAFC; 
        line-height: 1.2;
    }

    .rating-badge { 
        background-color: #F59E0B; 
        color: #FFFFFF; 
        padding: 4px 8px; 
        border-radius: 6px; 
        font-weight: 700; 
        font-size: 12px; 
        white-space: nowrap;
    }
    
    .hotel-location { 
        color: #38BDF8 !important; /* Sky Blue for high visibility */
        font-size: 13px; 
        font-weight: 500;
        margin-bottom: 12px; 
        display: block;
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis;
    }
    
    .hotel-desc { 
        font-size: 13.5px; 
        line-height: 1.5; 
        color: #CBD5E1; 
        flex-grow: 1; /* Pushes amenities to the bottom */
        overflow: hidden; 
        margin-bottom: 10px;
    }
    
    .hotel-facilities-wrapper {
        margin-top: auto; /* Locks to bottom */
        padding-top: 12px;
        border-top: 1px solid #334155; 
    }
    
    .hotel-facilities { 
        font-size: 12px; 
        color: #7DD3FC; 
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load Real Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hotels.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('hotels.csv', encoding='latin1')
    
    df = df.sample(n=10000, random_state=42)
    df.columns = df.columns.str.strip()
    df = df.fillna('')
    
    df['Search_Text'] = df['HotelName'].astype(str) + " " + \
                        df['Description'].astype(str) + " " + \
                        df['HotelFacilities'].astype(str) + " " + \
                        df['cityName'].astype(str) + " " + \
                        df['countyName'].astype(str)
    return df

# --- 2. Load NLP Model & FAISS ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_index(_model, text_data):
    embeddings = _model.encode(text_data, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Load Backend
with st.spinner("Initializing AI Core and loading dataset..."):
    df = load_data()
    model = load_model()
    if not df.empty:
        index = build_index(model, df['Search_Text'].tolist())

# --- UI CONTENT ---
st.title("🌍 Semantic Hotel Search Engine")

tab1, tab2, tab3 = st.tabs(["🏨 AI Search Agent", "📊 Internal Dataset", "💻 API Simulation"])

# ==========================================
# TAB 1: THE MAIN APP
# ==========================================
with tab1:
    st.markdown("Describe your ideal stay, and our NLP engine will find the perfect semantic match.")
    query = st.text_input("What kind of stay are you looking for?", placeholder="e.g., A quiet boutique hotel with a pool in Spain")

    if query and not df.empty:
        query_vector = model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        distances, indices = index.search(query_vector, k=len(df))
        results_df = df.iloc[indices[0]].copy()
        
        # Location Filtering Logic
        query_lower = query.lower()
        all_countries = [str(c).strip() for c in df['countyName'].unique() if str(c).strip()]
        detected_countries = [c for c in all_countries if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]
        all_cities = [str(c).strip() for c in df['cityName'].unique() if str(c).strip()]
        detected_cities = [c for c in all_cities if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]

        if detected_countries:
            results_df = results_df[results_df['countyName'].astype(str).str.contains('|'.join(detected_countries), case=False, na=False)]
            st.info(f"🌍 Auto-detected Country filter: **{', '.join(detected_countries)}**")
            
        if detected_cities:
            results_df = results_df[results_df['cityName'].astype(str).str.contains('|'.join(detected_cities), case=False, na=False)]
            st.info(f"🏙️ Auto-detected City filter: **{', '.join(detected_cities)}**")

        top_matches = results_df.head(12) 
        
        if top_matches.empty:
             st.warning("No hotels found matching that specific location. Try adjusting your search!")
        else:
            st.markdown("### ✨ Top Recommendations")
            cols = st.columns(3)
            
            for index, (_, hotel) in enumerate(top_matches.iterrows()):
                col = cols[index % 3] 
                
                with col:
                    # Data Sanitization
                    h_name = str(hotel['HotelName'])
                    # Remove hidden newlines and truncate for UI stability
                    raw_desc = str(hotel['Description']).replace('\n', ' ').replace('\r', ' ').strip()
                    h_desc = raw_desc[:180] + "..." if len(raw_desc) > 180 else raw_desc
                    
                    h_fac = str(hotel['HotelFacilities'])
                    h_rating = str(hotel.get('HotelRating', 'N/A'))
                    h_loc = f"{hotel.get('Address', 'Unknown')}, {hotel['cityName']}"
                    
                    st.markdown(f"""
                    <div class="hotel-card">
                        <div class="hotel-header">
                            <div class="hotel-name">{h_name}</div>
                            <div class="rating-badge">{h_rating}</div>
                        </div>
                        <div class="hotel-location">📍 {h_loc}</div>
                        <div class="hotel-desc">{h_desc}</div>
                        <div class="hotel-facilities-wrapper">
                            <div class="hotel-facilities">✨ <strong>Amenities:</strong> {h_fac}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2 & 3 Logic
# ==========================================
with tab2:
    st.metric(label="Total Hotels in Active Memory", value=f"{len(df):,}")
    available_columns = [col for col in ['HotelName', 'cityName', 'countyName', 'HotelRating'] if col in df.columns]
    st.dataframe(df[available_columns].head(100), use_container_width=True)

with tab3:
    if query:
        api_response = {
            "status": "success",
            "search_query": query,
            "filters": {"countries": detected_countries, "cities": detected_cities},
            "results": top_matches[['HotelName', 'cityName', 'countyName']].to_dict(orient="records")
        }
        st.json(api_response)
    else:
        st.info("Run a search to see the simulated API payload.")