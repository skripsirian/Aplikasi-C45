import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from c45 import DecisionTreeC45  # modul buatan sendiri

# === Plot Confusion Matrix ===
def plot_confusion_matrix(cm, classes=['Rendah', 'Tinggi']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    if not os.path.exists('temp'):
        os.makedirs('temp')
    plt.savefig('temp/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

# === Display Metrics Cards & Detail ===
def display_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=['Rendah', 'Tinggi'], output_dict=True)
    acc = (y_true == y_pred).mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    weighted_metrics = report.get('weighted avg', {})

    col1.metric("Accuracy", f"{acc:.2f}%")
    col2.metric("Precision", f"{weighted_metrics.get('precision', 0) * 100:.2f}%")
    col3.metric("Recall", f"{weighted_metrics.get('recall', 0) * 100:.2f}%")
    col4.metric("F1-Score", f"{weighted_metrics.get('f1-score', 0) * 100:.2f}%")

    st.markdown("### 📊 Metrik Detail per Kelas")

    class_metrics = []
    for class_name in ['Rendah', 'Tinggi']:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'Status': class_name,
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall': f"{metrics['recall']*100:.2f}%",
                'F1-Score': f"{metrics['f1-score']*100:.2f}%",
                'Support': int(metrics['support'])
            })

    if class_metrics:
        metrics_df = pd.DataFrame(class_metrics)

        def color_status(val):
            if val == 'Tinggi':
                return 'background-color: #28a745; color: white'
            elif val == 'Rendah':
                return 'background-color: #dc3545; color: white'
            return ''

        styled_df = metrics_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Data metrik per kelas tidak tersedia.")

# === Manual Input Form ===
def show_manual_input():
    st.info("Masukkan data siswa secara manual")

    # Initialize session state for storing inputs
    if 'input_data_list' not in st.session_state:
        st.session_state.input_data_list = []
    
    # Form for input
    with st.form("input_form"):
        nama_siswa = st.text_input("👤 Nama Siswa", placeholder="Masukkan nama siswa...")
        jenis_kelamin = st.text_input("👥 Jenis Kelamin", placeholder="Masukkan Laki-laki atau Perempuan...")
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.radio("Target Klasifikasi", ["Tinggi", "Rendah"])
        
        input_data = {
            'Nama': nama_siswa,
            'Jenis Kelamin': jenis_kelamin,
            'Keberagaman Teman': st.slider("Keberagaman Teman", 1, 5, 3),
            'Kemampuan Komunikasi': st.slider("Kemampuan Komunikasi", 1, 5, 3),
            'Empati & Pengertian': st.slider("Empati & Pengertian", 1, 5, 3),
            'Kerjasama & Kolaborasi': st.slider("Kerjasama & Kolaborasi", 1, 5, 3),
            'Mengelola Konflik': st.slider("Mengelola Konflik", 1, 5, 3),
            'Dukungan Sosial': st.slider("Dukungan Sosial", 1, 5, 3),
            'Kepemimpinan & Tanggung Jawab': st.slider("Kepemimpinan & Tanggung Jawab", 1, 5, 3),
            'target': target
        }

        submitted = st.form_submit_button("Tambah Data")
        if submitted:
            if not nama_siswa.strip():
                st.warning("⚠️ Mohon masukkan nama siswa terlebih dahulu!")
                return
            if not jenis_kelamin.strip():
                st.warning("⚠️ Mohon masukkan jenis kelamin siswa terlebih dahulu!")
                return
            if jenis_kelamin.strip() not in ["Laki-laki", "Perempuan"]:
                st.warning("⚠️ Jenis kelamin harus 'Laki-laki' atau 'Perempuan'!")
                return
            
            st.session_state.input_data_list.append(input_data)
            st.success(f"✅ Data untuk {nama_siswa} berhasil ditambahkan!")

    # Show collected data
    if st.session_state.input_data_list:
        st.markdown("### 📊 Data yang Terkumpul")
        df_collected = pd.DataFrame(st.session_state.input_data_list)
        st.dataframe(df_collected, use_container_width=True)
        
        # Show class distribution
        st.markdown("### 📊 Distribusi Target")
        class_dist = df_collected['target'].value_counts()
        st.write(pd.DataFrame({
            'Kelas': class_dist.index,
            'Jumlah': class_dist.values,
            'Persentase': [f"{(val/len(df_collected)*100):.2f}%" for val in class_dist.values]
        }))

        # Add button to clear data
        if 'show_delete_confirmation' not in st.session_state:
            st.session_state.show_delete_confirmation = False

        if not st.session_state.show_delete_confirmation:
            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.show_delete_confirmation = True
                st.rerun()
        
        if st.session_state.show_delete_confirmation:
            st.warning("⚠️ Apakah Anda yakin ingin menghapus semua data?", icon="⚠️")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Ya, Hapus Data"):
                    st.session_state.input_data_list = []
                    st.rerun()
            with col2:
                if st.button("❌ Tidak, Batalkan"):
                    st.rerun()

        # Only show prediction button if we have enough data
        if len(df_collected) >= 4 and len(class_dist) >= 2:
            if st.button("🔍 Jalankan Klasifikasi", type="primary"):
                # Prepare data for training
                X = df_collected.drop(['target', 'Nama', 'Jenis Kelamin'], axis=1)
                y = df_collected['target']

                # Split data with stratification if possible
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=0.3, 
                        random_state=42,
                        stratify=y
                    )
                except ValueError:
                    st.warning("⚠️ Tidak bisa melakukan stratified split, menggunakan random split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=0.3, 
                        random_state=42
                    )

                model = DecisionTreeC45()
                model.fit(X_train, y_train)

                # Make predictions
                test_preds = model.predict(X_test)
                
                st.markdown("### 📈 Evaluasi Model")
                display_metrics(y_test, test_preds)

                cm = confusion_matrix(y_test, test_preds, labels=['Rendah', 'Tinggi'])
                plot_confusion_matrix(cm)
                st.image('temp/confusion_matrix.png', use_container_width=True)

                # Show test results
                st.markdown("### 🔍 Detail Hasil Testing")
                test_data = df_collected.iloc[X_test.index]
                results_df = pd.DataFrame({
                    'Nama': test_data['Nama'],
                    'Jenis Kelamin': test_data['Jenis Kelamin'],
                    'Aktual': y_test,
                    'Prediksi': test_preds,
                    'Benar/Salah': y_test == test_preds
                })
                st.dataframe(results_df, use_container_width=True)

        else:
            min_samples_needed = max(4 - len(df_collected), 0)
            if len(class_dist) < 2:
                st.warning("⚠️ Dibutuhkan minimal 1 data untuk setiap kelas (Tinggi dan Rendah)!")
            else:
                st.warning(f"⚠️ Tambahkan minimal {min_samples_needed} data lagi untuk melakukan klasifikasi!")

# === Main App ===
def main():
    st.set_page_config(page_title="Aplikasi C4.5 - Pola Pertemanan", layout="wide")
    st.title("📊 Aplikasi C4.5 - Klasifikasi Pola Pertemanan Siswa")
    show_manual_input()

if __name__ == "__main__":
    main()
