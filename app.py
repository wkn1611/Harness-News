import streamlit as st
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

# Load cấu hình
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "hermes_news")

# Thiết lập trang - Giao diện rộng và tiêu đề trình duyệt
st.set_page_config(page_title="Hermes AI News", layout="wide", page_icon="⚡")

# --- CSS CUSTOM: Biến Streamlit thành một bản tin hiện đại ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    /* Thiết kế Card bài viết */
    .news-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .news-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    .news-title {
        color: #58a6ff;
        font-size: 1.4rem;
        font-weight: 800;
        text-decoration: none;
        margin-bottom: 10px;
        display: block;
    }
    
    .news-summary {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 15px 0;
    }
    
    .tag-container {
        display: flex;
        gap: 8px;
        margin-top: 10px;
    }
    
    .tag {
        background-color: #21262d;
        color: #8b949e;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #30363d;
    }
    
    .meta-data {
        color: #8b949e;
        font-size: 0.8rem;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE LOGIC ---
async def fetch_news(tag_filter=None, search_query=None):
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db["articles"]
    
    query = {}
    if tag_filter and tag_filter != "Tất cả":
        query["tags"] = tag_filter
    if search_query:
        query["title"] = {"$regex": search_query, "$options": "i"}
        
    cursor = collection.find(query).sort("created_at", -1).limit(50)
    return await cursor.to_list(length=50)

# --- UI RENDER ---
def main():
    st.title("⚡ Hermes AI News")
    st.markdown("---")

    # Sidebar: Bộ lọc
    with st.sidebar:
        st.header("Bộ lọc tin tức")
        search = st.text_input("🔍 Tìm kiếm tiêu đề...")
        # Giả lập danh sách tags, bạn có thể lấy từ DB nếu muốn
        tag_list = ["Tất cả", "AI", "Programming", "Hardware", "Python", "LLM"]
        category = st.selectbox("Chuyên mục", tag_list)
        
        st.markdown("---")
        st.info("Hệ thống tự động cập nhật lúc 08:00, 14:00 và 20:00 hàng ngày.")

    # Fetch dữ liệu
    articles = asyncio.run(fetch_news(category, search))

    if not articles:
        st.warning("Không tìm thấy bài viết nào phù hợp.")
        return

    # Hiển thị dạng Grid (2 cột)
    col1, col2 = st.columns(2)
    
    for idx, art in enumerate(articles):
        target_col = col1 if idx % 2 == 0 else col2
        
        with target_col:
            # Render Card bằng HTML
            tags_html = "".join([f'<span class="tag">{t}</span>' for t in art.get('tags', [])])
            date_str = art.get('created_at', datetime.now()).strftime("%d/%m/%Y %H:%M")
            
            st.markdown(f"""
            <div class="news-card">
                <div class="meta-data">🕒 {date_str}</div>
                <a href="{art['url']}" target="_blank" class="news-title">{art['title']}</a>
                <div class="news-summary">
                    {art.get('summary', 'Đang cập nhật tóm tắt...')}
                </div>
                <div class="tag-container">
                    {tags_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()