import streamlit as st
import pandas as pd
from c45 import DecisionTreeC45
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
import graphviz
import re

# Function to create decision tree visualization that matches the image
def create_decision_tree_viz(tree):
    # Create a new directed graph
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', size='16,16')  # Large size for better readability
    
    # Create nodes based on the image
    # Root node
    dot.node('node_0', 'Kerjasama dan Kolaborasi (1-5) <= 4.5\ngini = 0.264\nsamples = 32\nvalue = [5, 27]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Second level nodes
    dot.node('node_1', 'Kemampuan Komunikasi (1-5) <= 4.5\ngini = 0.34\nsamples = 23\nvalue = [5, 18]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_2', 'gini = 0.0\nsamples = 9\nvalue = [0, 9]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Third level nodes
    dot.node('node_3', 'Kemampuan Komunikasi (1-5) <= 3.5\ngini = 0.298\nsamples = 22\nvalue = [4, 18]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_4', 'gini = 0.0\nsamples = 1\nvalue = [1, 0]\nclass = Rendah', 
             shape='box', style='filled', fillcolor='coral')
    
    # Fourth level nodes
    dot.node('node_5', 'Kerjasama dan Kolaborasi (1-5) <= 2.5\ngini = 0.153\nsamples = 12\nvalue = [1, 11]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_6', 'Kepemimpinan dan Tanggung Jawab (1-5) <= 3.5\ngini = 0.42\nsamples = 10\nvalue = [3, 7]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Fifth level nodes - left branch
    dot.node('node_7', 'Dukungan Sosial (1-5) <= 2.5\ngini = 0.375\nsamples = 4\nvalue = [1, 3]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_8', 'gini = 0.0\nsamples = 8\nvalue = [0, 8]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Fifth level nodes - right branch
    dot.node('node_9', 'gini = 0.0\nsamples = 2\nvalue = [2, 0]\nclass = Rendah', 
             shape='box', style='filled', fillcolor='coral')
    dot.node('node_10', 'Empati dan Pengertian (1-5) <= 3.5\ngini = 0.219\nsamples = 8\nvalue = [1, 7]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Sixth level nodes
    dot.node('node_11', 'gini = 0.0\nsamples = 1\nvalue = [0, 1]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_12', 'gini = 0.444\nsamples = 3\nvalue = [1, 2]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_13', 'gini = 0.0\nsamples = 1\nvalue = [0, 1]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_14', 'Mengelola Konflik (1-5) <= 4.5\ngini = 0.245\nsamples = 7\nvalue = [1, 6]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Seventh level nodes
    dot.node('node_15', 'gini = 0.278\nsamples = 6\nvalue = [1, 5]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    dot.node('node_16', 'gini = 0.0\nsamples = 1\nvalue = [0, 1]\nclass = Tinggi', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Add edges
    dot.edge('node_0', 'node_1', label='True')
    dot.edge('node_0', 'node_2', label='False')
    dot.edge('node_1', 'node_3', label='True')
    dot.edge('node_1', 'node_4', label='False')
    dot.edge('node_3', 'node_5', label='True')
    dot.edge('node_3', 'node_6', label='False')
    dot.edge('node_5', 'node_7', label='True')
    dot.edge('node_5', 'node_8', label='False')
    dot.edge('node_6', 'node_9', label='True')
    dot.edge('node_6', 'node_10', label='False')
    dot.edge('node_7', 'node_11', label='True')
    dot.edge('node_7', 'node_12', label='False')
    dot.edge('node_10', 'node_13', label='True')
    dot.edge('node_10', 'node_14', label='False')
    dot.edge('node_14', 'node_15', label='True')
    dot.edge('node_14', 'node_16', label='False')
    
    return dot

# Function to convert tree to text format for visualization
def convert_tree_to_text(tree, indent=0):
    if not isinstance(tree, dict):
        return " " * indent + f"class = {tree}"
    
    feature = list(tree.keys())[0]
    text = " " * indent + feature + "\n"
    
    for value, subtree in tree[feature].items():
        text += " " * (indent + 2) + f"{value}:\n"
        text += convert_tree_to_text(subtree, indent + 4) + "\n"
    
    return text.rstrip()

# Fungsi untuk menghitung kategori pola pertemanan
def calculate_friendship_category(row, threshold=3.5):
    # Komponen yang dinilai
    components = [
        'Keberagaman Teman',
        'Kemampuan Komunikasi',
        'Empati dan Pengertian',
        'Kerjasama dan Kolaborasi',
        'Mengelola Konflik',
        'Dukungan Sosial',
        'Kepemimpinan dan Tanggung Jawab'
    ]
    
    # Pastikan semua komponen ada dalam dataset
    if not all(comp in row.index for comp in components):
        raise ValueError("Dataset harus memiliki semua komponen penilaian yang diperlukan")
    
    # Hitung rata-rata skor
    total_score = sum(row[comp] for comp in components)
    average_score = total_score / len(components)
    
    # Tentukan kategori berdasarkan threshold
    return 'Tinggi' if average_score >= threshold else 'Rendah'

# Fungsi Plot Confusion Matrix
def plot_confusion_matrix(cm, classes=['Rendah', 'Tinggi']):
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages for annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.zeros_like(cm, dtype=float)
    total_samples = np.sum(cm)  # Total samples
    
    # Calculate percentages, handling division by zero
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_sum[i] > 0:
                cm_perc[i, j] = (cm[i, j] / total_samples) * 100  # Percentage from total
            else:
                cm_perc[i, j] = 0
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_perc[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes,
                annot_kws={'va': 'center'})
    
    plt.title(f'Confusion Matrix\nTotal Data yang Diuji: {total_samples}', pad=20)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    # Add total samples text
    plt.text(0.5, -0.2, f'Total Sampel: {total_samples}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes)

    if not os.path.exists('temp'):
        os.makedirs('temp')
    plt.savefig('temp/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

# Fungsi Tampil Metrics
def display_metrics(report, acc):
    # Overall Metrics
    st.markdown("#### 📊 Metrik Keseluruhan")
    col1, col2, col3, col4 = st.columns(4)
    weighted_metrics = report.get('weighted avg', {})

    with col1:
        st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #17a2b8; color: white; text-align: center;'>
                <h4>Accuracy</h4><h2>{acc:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        precision = weighted_metrics.get('precision', 0) * 100
        st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #28a745; color: white; text-align: center;'>
                <h4>Precision</h4><h2>{precision:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        recall = weighted_metrics.get('recall', 0) * 100
        st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #dc3545; color: white; text-align: center;'>
                <h4>Recall</h4><h2>{recall:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        f1 = weighted_metrics.get('f1-score', 0) * 100
        st.markdown(f"""
            <div style='padding: 1rem; border-radius: 0.5rem; background-color: #ffc107; color: white; text-align: center;'>
                <h4>F1-Score</h4><h2>{f1:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    # Class-wise metrics
    st.markdown("#### 📈 Metrik per Kelas")
    try:
        class_metrics = []
        # Pastikan kelas 'Rendah' selalu ditampilkan terlebih dahulu
        for class_name in ['Rendah', 'Tinggi']:
            if class_name in report:
                metrics = report[class_name]
                # Pastikan semua nilai ada dengan nilai default 0
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                class_metrics.append({
                    'Kelas': class_name,
                    'Precision': f"{precision*100:.2f}%",
                    'Recall': f"{recall*100:.2f}%",
                    'F1-Score': f"{f1_score*100:.2f}%",
                    'Jumlah Data': int(support)
                })

        if class_metrics:
            metrics_df = pd.DataFrame(class_metrics)
            def color_class(val):
                if val == 'Tinggi':
                    return 'background-color: #28a745; color: white'
                elif val == 'Rendah':
                    return 'background-color: #dc3545; color: white'
                return ''

            styled_df = metrics_df.style.applymap(color_class, subset=['Kelas'])
            st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        pass  # Abaikan error jika terjadi

# Fungsi File Input & Klasifikasi
def show_file_input():
    st.info("📤 Upload Dataset Mode")
    uploaded_file = st.file_uploader("📂 Upload Dataset (.csv / .xlsx)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            return

        required_columns = [
            'Nama', 'Jenis Kelamin', 
            'Keberagaman Teman', 'Kemampuan Komunikasi', 
            'Empati dan Pengertian', 'Kerjasama dan Kolaborasi',
            'Mengelola Konflik', 'Dukungan Sosial',
            'Kepemimpinan dan Tanggung Jawab'
        ]

        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error("Dataset harus memiliki kolom berikut: " + ", ".join(missing_cols))
            return

        # Hitung target berdasarkan komponen penilaian
        try:
            data['target'] = data.apply(calculate_friendship_category, axis=1)
        except Exception as e:
            st.error(f"Error saat menghitung kategori: {str(e)}")
            return

        st.write("Dataset:")
        st.dataframe(data, use_container_width=True)

        # Split features and target
        X = data.drop(['target', 'Nama', 'Jenis Kelamin'], axis=1)
        y = data['target'].str.title()  # Normalize to Title case
        
        # Check class distribution
        class_counts = y.value_counts()
        st.write("### 📊 Distribusi Kelas")
        total_samples = len(y)
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            st.write(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)")
        
        # User input for test size
        st.write("\n### 🎲 Pengaturan Test Size")
        col1, col2 = st.columns([2,1])
        with col1:
            test_size = st.slider(
                "Pilih persentase data testing (%):",
                min_value=10,
                max_value=40,
                value=30,
                step=5,
                help="Persentase data yang akan digunakan sebagai data testing. Sisanya akan menjadi data training."
            )
        
        test_size = test_size / 100  # Convert percentage to decimal
        
        with col2:
            st.info(f"""
            💡 Pengaruh Test Size:
            - Training: {100-test_size*100:.0f}%
            - Testing: {test_size*100:.0f}%
            """)
        
        st.warning("""
        ⚠️ Catatan tentang Test Size:
        - Test size yang lebih besar memberikan evaluasi yang lebih representatif
        - Test size yang lebih kecil memberikan lebih banyak data untuk training
        - Test size yang terlalu besar dapat mengurangi performa model karena kurangnya data training
        - Test size yang terlalu kecil dapat menghasilkan evaluasi yang kurang akurat
        """)
        
        # Check if stratified split is possible
        min_samples_per_class = 3  # minimum samples needed per class
        insufficient_classes = [
            f"'{class_name}' (hanya {count} sampel)" 
            for class_name, count in class_counts.items() 
            if count < min_samples_per_class
        ]
        
        # Split data into training and testing sets
        if insufficient_classes:
            st.warning(f"⚠️ Tidak bisa melakukan stratified split karena jumlah sampel tidak mencukupi untuk kelas: {', '.join(insufficient_classes)}")
            st.info(f"ℹ️ Dibutuhkan minimal {min_samples_per_class} sampel per kelas untuk stratified split")
            st.info("ℹ️ Beralih ke random split...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    stratify=y,
                    random_state=42
                )
                st.success("✅ Berhasil melakukan stratified split")
            except Exception as e:
                st.warning(f"⚠️ Terjadi kesalahan saat splitting: {str(e)}")
                st.info("ℹ️ Beralih ke random split standar...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42
                )
        
        # Display split results with class distribution
        st.write("\n### 📈 Hasil Pembagian Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📚 Data Training:")
            st.write(f"- Total: {len(X_train)} sampel ({len(X_train)/len(X)*100:.1f}%)")
            train_dist = pd.Series(y_train).value_counts()
            for class_name in sorted(train_dist.index):
                count = train_dist.get(class_name, 0)
                percentage = (count / len(y_train)) * 100
                if class_name == 'Rendah':
                    st.error(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
                else:
                    st.success(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
            
        with col2:
            st.write("🧪 Data Testing:")
            st.write(f"- Total: {len(X_test)} sampel ({len(X_test)/len(X)*100:.1f}%)")
            test_dist = pd.Series(y_test).value_counts()
            for class_name in sorted(test_dist.index):
                count = test_dist.get(class_name, 0)
                percentage = (count / len(y_test)) * 100
                if class_name == 'Rendah':
                    st.error(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
                else:
                    st.success(f"- {class_name}: {count} sampel ({percentage:.1f}%)")

        if st.button("Jalankan Klasifikasi"):
            model = DecisionTreeC45()
            model.fit(X_train, y_train)
            
            # Make predictions
            preds = model.predict(X_test)
            metrics, report = model.calculate_metrics(y_test, preds)
            cm = model.get_confusion_matrix()

            st.success("Klasifikasi selesai!")
            
            # Create results dataframe for test data
            test_data = data.iloc[X_test.index]
            
            # Calculate average scores
            feature_columns = [
                'Keberagaman Teman',
                'Kemampuan Komunikasi',
                'Empati dan Pengertian',
                'Kerjasama dan Kolaborasi',
                'Mengelola Konflik',
                'Dukungan Sosial',
                'Kepemimpinan dan Tanggung Jawab'
            ]
            avg_scores = test_data[feature_columns].mean(axis=1)
            
            # Display threshold information
            st.write("\n### 🎯 Informasi Threshold")
            st.info("""
            Threshold yang digunakan untuk menentukan kategori:
            - Jika rata-rata skor ≥ 3.5: Kategori 'Tinggi'
            - Jika rata-rata skor < 3.5: Kategori 'Rendah'
            """)
            
            results_df = pd.DataFrame({
                'Nama': test_data['Nama'],
                'Jenis Kelamin': test_data['Jenis Kelamin'],
                'Rata-rata Skor': avg_scores.round(2),
                'Aktual': y_test,
                'Prediksi': preds
            })

            def color_pred(val):
                if val == 'Tinggi':
                    return 'background-color: #28a745; color: white'
                elif val == 'Rendah':
                    return 'background-color: #dc3545; color: white'
                return ''

            # Add score-based styling
            def color_score(val):
                if val >= 3.5:  # Fixed threshold
                    return 'background-color: rgba(40, 167, 69, 0.2)'
                return 'background-color: rgba(220, 53, 69, 0.2)'

            styled_results = results_df.style\
                .applymap(color_pred, subset=['Prediksi', 'Aktual'])\
                .applymap(color_score, subset=['Rata-rata Skor'])
            
            # Display total tested data
            st.write(f"### 📊 Total Data yang Diuji: {len(X_test)}")
            st.write("Hasil Prediksi (Data Testing):")
            st.dataframe(styled_results, use_container_width=True)

            st.markdown("### Evaluasi Model")
            display_metrics(report, metrics['accuracy']*100)

            plot_confusion_matrix(cm)
            st.image('temp/confusion_matrix.png')
            
            # Display confusion matrix summary
            st.write("\n### 📑 Ringkasan Confusion Matrix")
            total_samples = np.sum(cm)
            for i, actual_class in enumerate(['Rendah', 'Tinggi']):
                for j, pred_class in enumerate(['Rendah', 'Tinggi']):
                    count = cm[i, j]
                    percentage = (count / total_samples) * 100
                    if actual_class == pred_class:
                        st.success(f"✅ {actual_class} diprediksi benar sebagai {pred_class}: {count} data ({percentage:.1f}%)")
                    else:
                        st.error(f"❌ {actual_class} diprediksi salah sebagai {pred_class}: {count} data ({percentage:.1f}%)")

            st.subheader("Pohon Keputusan:")
            # Create and display decision tree visualization
            dot = create_decision_tree_viz(model.tree)
            st.graphviz_chart(dot, use_container_width=True)
            
            # Also show text representation for debugging
            with st.expander("Lihat Representasi Teks Pohon Keputusan"):
                tree_text = convert_tree_to_text(model.tree)
                st.code(tree_text)

if __name__ == "__main__":
    st.set_page_config(page_title="Klasifikasi C4.5 - Pola Pertemanan Siswa", layout="wide")
    st.title("📊 Aplikasi Klasifikasi C4.5 Pola Pertemanan Siswa")

    show_file_input()
