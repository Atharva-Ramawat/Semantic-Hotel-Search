# frontend_streamlit/app.py
import streamlit as st
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, CSS_STYLE

# Setup MUST be the very first Streamlit command
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Import our custom modules
from data_loader import load_data
from model import load_nlp_model, build_index
from search import execute_search
from sidebar import render_sidebar

# Initialize Data & Models
with st.spinner("Initializing AI Core and loading dataset..."):
    df = load_data()
    model = load_nlp_model()
    if not df.empty:
        index = build_index(model, df['Search_Text'].tolist())

# Render the UI
render_sidebar()
st.title("🌍 Semantic Hotel Search Engine")
tab1, tab2, tab3 = st.tabs(["🏨 AI Search Agent", "📊 Internal Dataset", "💻 API Simulation"])

with tab1:
    st.markdown("Describe your ideal stay, and our NLP engine will find the perfect semantic match.")
    query = st.text_input("What kind of stay are you looking for?", placeholder="e.g., A quiet boutique hotel with a pool in Spain")

    if query and not df.empty:
        top_matches, detected_countries, detected_cities = execute_search(query, df, model, index)

        if detected_countries:
            st.info(f"🌍 Auto-detected Country filter: **{', '.join(detected_countries)}**")
        if detected_cities:
            st.info(f"🏙️ Auto-detected City filter: **{', '.join(detected_cities)}**")

        if top_matches.empty:
             st.warning("No hotels found matching that specific location. Try adjusting your search!")
        else:
            st.markdown("### ✨ Top Recommendations")
            cols = st.columns(3)
            
            for index, (_, hotel) in enumerate(top_matches.iterrows()):
                col = cols[index % 3] 
                with col:
                    h_name = str(hotel['HotelName'])
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

with tab2:
    st.metric(label="Total Hotels in Active Memory", value=f"{len(df):,}")
    available_cols = [col for col in ['HotelName', 'cityName', 'countyName', 'HotelRating'] if col in df.columns]
    st.dataframe(df[available_cols].head(100), use_container_width=True)

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