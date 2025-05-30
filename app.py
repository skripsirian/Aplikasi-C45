import streamlit as st
from manual_input import show_manual_input
from file_input import show_file_input

# ------------------ Styling ------------------
st.set_page_config(page_title="Pola Pertemanan Siswa - C4.5", page_icon="🤝", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content .block-container {
        padding: 2rem 1rem;
    }
    .menu-button {
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        cursor: pointer;
        transition: all 0.3s;
    }
    .menu-button:hover {
        background-color: #e9ecef;
        border-color: #dee2e6;
    }
    .menu-button.active {
        background-color: #007bff;
        color: white;
        border-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤝 Pola Pertemanan Siswa dengan Algoritma C4.5")
st.subheader("📊 Klasifikasi Kecerdasan Sosial & Emosional")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Menu")
    st.markdown("---")
    
    # Create menu buttons with unique keys
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Upload Dataset", key="upload_dataset_btn", use_container_width=True):
            st.session_state.mode = "Upload Dataset"
    with col2:
        if st.button("✏️ Input Manual", key="input_manual_btn", use_container_width=True):
            st.session_state.mode = "Input Manual"
            
    if 'mode' not in st.session_state:
        st.session_state.mode = "Upload Dataset"
        
    st.markdown(f"**Mode Aktif:** {st.session_state.mode}")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Informasi
    - Upload dataset dalam format CSV/Excel
    - Pastikan ada kolom 'target' dengan nilai 'Tinggi'/'Rendah'
    - Semua nilai harus dalam skala 1-5
    """)

# ------------------ Main Content ------------------
if st.session_state.mode == "Input Manual":
    show_manual_input()
else:  # Upload Dataset mode
    show_file_input()
