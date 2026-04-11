# frontend_streamlit/config.py
PAGE_TITLE = "AI Travel Agent"
PAGE_ICON = "🌍"
LAYOUT = "wide"

CSS_STYLE = """
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    h1, h2, h3 { color: #38BDF8 !important; font-weight: 600; letter-spacing: -0.5px; }
    
    .hotel-card { 
        background-color: #1E293B; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #334155;
        border-left: 5px solid #38BDF8; 
        margin-bottom: 20px; 
        height: 340px; 
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .hotel-card:hover { transform: translateY(-4px); border-left: 5px solid #7DD3FC; }
    
    .hotel-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
    .hotel-name { font-size: 18px; font-weight: 700; color: #F8FAFC; line-height: 1.2; }
    .rating-badge { background-color: #F59E0B; color: #FFFFFF; padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .hotel-location { color: #38BDF8 !important; font-size: 13px; font-weight: 500; margin-bottom: 12px; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .hotel-desc { font-size: 13.5px; line-height: 1.5; color: #CBD5E1; flex-grow: 1; overflow: hidden; margin-bottom: 10px; }
    .hotel-facilities-wrapper { margin-top: auto; padding-top: 12px; border-top: 1px solid #334155; }
    .hotel-facilities { font-size: 12px; color: #7DD3FC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
"""