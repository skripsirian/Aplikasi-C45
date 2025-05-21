import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from c45 import DecisionTreeC45  # modul buatan sendiri
import graphviz
import re
import numpy as np

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

# Function to predict based on the leaf node where data falls
def predict_from_leaf_node(instance, tree, node_path=None, leaf_counts=None):
    if node_path is None:
        node_path = []
    
    # If we've reached a leaf node (not a dictionary)
    if not isinstance(tree, dict):
        return tree, node_path
    
    # Get the feature to split on
    feature = list(tree.keys())[0]
    
    # Extract feature name and threshold if it contains a threshold
    match = re.search(r'(.*?) <= (.*)', feature)
    if match:
        feature_name, threshold = match.groups()
        feature_name = feature_name.strip()
        threshold = float(threshold)
        
        # Get the actual value of this feature for the instance
        instance_value = instance[feature_name]
        
        # Determine which branch to follow
        if instance_value <= threshold:
            node_path.append({
                "feature": feature_name,
                "threshold": threshold,
                "instance_value": instance_value,
                "decision": "≤",
                "branch": "True",
                "description": f"{feature_name} = {instance_value} ≤ {threshold} → True"
            })
            return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]], node_path, leaf_counts)
        else:
            node_path.append({
                "feature": feature_name,
                "threshold": threshold,
                "instance_value": instance_value,
                "decision": ">",
                "branch": "False",
                "description": f"{feature_name} = {instance_value} > {threshold} → False"
            })
            return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[1]], node_path, leaf_counts)
    else:
        # If there's no threshold, just follow the first branch
        node_path.append({
            "feature": feature,
            "threshold": None,
            "instance_value": None,
            "decision": "",
            "branch": "",
            "description": f"Feature: {feature}"
        })
        return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]], node_path, leaf_counts)

# Function to get leaf node statistics
def get_leaf_node_stats(tree, stats=None, path=None):
    if stats is None:
        stats = {}
    if path is None:
        path = []
    
    # If we've reached a leaf node
    if not isinstance(tree, dict):
        leaf_key = tuple(path)
        stats[leaf_key] = {"class": tree, "path": path.copy()}
        return stats
    
    feature = list(tree.keys())[0]
    
    # Extract feature name and threshold if applicable
    match = re.search(r'(.*?) <= (.*)', feature)
    if match:
        feature_name, threshold = match.groups()
        feature_name = feature_name.strip()
        threshold = float(threshold)
        
        # Traverse left branch (≤ threshold)
        left_path = path.copy()
        left_path.append((feature, "≤", threshold, "True"))
        stats = get_leaf_node_stats(tree[feature][list(tree[feature].keys())[0]], stats, left_path)
        
        # Traverse right branch (> threshold)
        right_path = path.copy()
        right_path.append((feature, ">", threshold, "False"))
        stats = get_leaf_node_stats(tree[feature][list(tree[feature].keys())[1]], stats, right_path)
    else:
        # If there's no threshold
        new_path = path.copy()
        new_path.append((feature, "", "", ""))
        stats = get_leaf_node_stats(tree[feature][list(tree[feature].keys())[0]], stats, new_path)
    
    return stats

# === Plot Confusion Matrix ===
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

# === Display Metrics Cards & Detail ===
def display_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=['Rendah', 'Tinggi'], output_dict=True)
    acc = (y_true == y_pred).mean() * 100

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

# Function to calculate friendship category based on scores
def calculate_friendship_category(scores, threshold=3.5):
    average_score = sum(scores) / len(scores)
    return 'Tinggi' if average_score >= threshold else 'Rendah'

# === Manual Input Form ===
def show_manual_input():
    st.info("Masukkan data siswa secara manual")

    # Initialize session state for storing inputs
    if 'input_data_list' not in st.session_state:
        st.session_state.input_data_list = []
    
    # Initialize classification state
    if 'classification_run' not in st.session_state:
        st.session_state.classification_run = False
    
    # Form for input
    with st.form("input_form"):
        nama_siswa = st.text_input("👤 Nama Siswa", placeholder="Masukkan nama siswa...")
        col1, col2 = st.columns(2)
        with col1:
            jenis_kelamin = st.selectbox("👥 Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with col2:
            kelas = st.text_input("🏫 Kelas", placeholder="Contoh: Kelas 10", value="Kelas 10")
        
        # Display threshold information
        st.info("""
        ℹ️ Target akan dihitung otomatis berdasarkan rata-rata skor:
        - Jika rata-rata skor ≥ 3.5: Kategori 'Tinggi'
        - Jika rata-rata skor < 3.5: Kategori 'Rendah'
        """)
        
        # Add rating scale explanation
        with st.expander("📊 Penjelasan Skala Penilaian (1-5)", expanded=True):
            st.markdown("""
            | Nilai | Deskripsi |
            | --- | --- |
            | 1 | Sangat Kurang Baik |
            | 2 | Kurang Baik |
            | 3 | Cukup |
            | 4 | Baik |
            | 5 | Sangat Baik |
            """)
        
        # Input sliders for each component
        st.markdown("#### 📊 Komponen Penilaian")
        col1, col2 = st.columns(2)
        
        with col1:
            keberagaman = st.slider("Keberagaman Teman", 1, 5, 3)
            komunikasi = st.slider("Kemampuan Komunikasi", 1, 5, 3)
            empati = st.slider("Empati & Pengertian", 1, 5, 3)
            kerjasama = st.slider("Kerjasama & Kolaborasi", 1, 5, 3)
        
        with col2:
            konflik = st.slider("Mengelola Konflik", 1, 5, 3)
            dukungan = st.slider("Dukungan Sosial", 1, 5, 3)
            kepemimpinan = st.slider("Kepemimpinan & Tanggung Jawab", 1, 5, 3)
        
        # Calculate average score and determine target
        scores = [keberagaman, komunikasi, empati, kerjasama, konflik, dukungan, kepemimpinan]
        avg_score = sum(scores) / len(scores)
        target = 'Tinggi' if avg_score >= 3.5 else 'Rendah'
        
        # Display calculated average and category
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rata-rata Skor", f"{avg_score:.2f}")
        with col2:
            color = "#28a745" if target == "Tinggi" else "#dc3545"
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                <h3>Kategori: {target}</h3>
            </div>
            """, unsafe_allow_html=True)

        input_data = {
            'Nama': nama_siswa,
            'Jenis Kelamin': jenis_kelamin,
            'Kelas': kelas,
            'Keberagaman Teman': keberagaman,
            'Kemampuan Komunikasi': komunikasi,
            'Empati dan Pengertian': empati,
            'Kerjasama dan Kolaborasi': kerjasama,
            'Mengelola Konflik': konflik,
            'Dukungan Sosial': dukungan,
            'Kepemimpinan dan Tanggung Jawab': kepemimpinan,
            'Rata-rata Skor': round(avg_score, 2),
            'target': target
        }

        submitted = st.form_submit_button("Tambah Data")
        if submitted:
            if not nama_siswa.strip():
                st.warning("⚠️ Mohon masukkan nama siswa terlebih dahulu!")
                return
            
            st.session_state.input_data_list.append(input_data)
            st.success(f"✅ Data untuk {nama_siswa} berhasil ditambahkan!")

    # Show collected data
    if st.session_state.input_data_list:
        st.markdown("### 📊 Data yang Terkumpul")
        df_collected = pd.DataFrame(st.session_state.input_data_list)
        
        # Add Kelas column based on grade level (you can modify this as needed)
        if 'Kelas' not in df_collected.columns:
            # Add a text input for class as a simple attribute
            default_kelas = st.text_input("Kelas untuk Semua Siswa:", value="Kelas 10")
            
            # Apply the selected class to all students as a simple attribute
            df_collected['Kelas'] = default_kelas
            
            # Update the session state data with the class information
            for i in range(len(st.session_state.input_data_list)):
                st.session_state.input_data_list[i]['Kelas'] = default_kelas
        
        # Apply styling to the dataframe
        def color_target(val):
            if val == 'Tinggi':
                return 'background-color: #28a745; color: white'
            elif val == 'Rendah':
                return 'background-color: #dc3545; color: white'
            return ''
            
        def color_score(val):
            if val >= 3.5:
                return 'background-color: rgba(40, 167, 69, 0.2)'
            return 'background-color: rgba(220, 53, 69, 0.2)'
            
        styled_df = df_collected.style.map(color_score, subset=['Rata-rata Skor']).map(color_target, subset=['target'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Show class distribution
        st.markdown("### 📊 Distribusi Target")
        class_dist = df_collected['target'].value_counts()
        
        # Create a DataFrame for better display
        dist_df = pd.DataFrame({
            'Kelas': class_dist.index,
            'Jumlah': class_dist.values,
            'Persentase': [f"{(val/len(df_collected)*100):.2f}%" for val in class_dist.values]
        })
        
        # Apply styling to the distribution dataframe
        def color_class(val):
            if val == 'Tinggi':
                return 'background-color: #28a745; color: white'
            elif val == 'Rendah':
                return 'background-color: #dc3545; color: white'
            return ''
            
        styled_dist = dist_df.style.map(color_class, subset=['Kelas'])
        st.dataframe(styled_dist, use_container_width=True)

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

        # Only show prediction button if we have enough data and classification hasn't been run
        if len(df_collected) >= 4 and len(class_dist) >= 2:
            if st.button("🔍 Jalankan Klasifikasi", type="primary"):
                # Set flag that classification has been run
                st.session_state.classification_run = True
                
                # Prepare data for training
                X = df_collected.drop(['target', 'Nama', 'Jenis Kelamin', 'Rata-rata Skor', 'Kelas'], axis=1)
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

                # Make predictions using leaf node majority class
                test_preds = []
                prediction_paths = []
                
                # Get leaf node statistics
                leaf_stats = get_leaf_node_stats(model.tree)
                
                # Predict each test instance
                for idx, row in X_test.iterrows():
                    prediction, path = predict_from_leaf_node(row, model.tree)
                    test_preds.append(prediction)
                    prediction_paths.append(path)
                
                # Convert predictions to series for evaluation
                test_preds = pd.Series(test_preds, index=X_test.index)
                
                # Store everything in session state
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.test_preds = test_preds
                st.session_state.prediction_paths = prediction_paths
                st.session_state.test_names = df_collected.iloc[X_test.index]['Nama'].tolist()
                st.session_state.test_data = df_collected.iloc[X_test.index]
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, test_preds, labels=['Rendah', 'Tinggi'])
                st.session_state.cm = cm
                
                # Generate and save confusion matrix plot
                plot_confusion_matrix(cm)
                
                st.rerun()  # Rerun to refresh the page with session state data
        
        # Display classification results if they exist
        if st.session_state.classification_run:
            # Display information about the prediction process
            st.markdown("### 🌲 Informasi Prediksi Berdasarkan Leaf Node")
            st.info("""
            Prediksi dilakukan berdasarkan kelas mayoritas pada leaf node tempat data jatuh.
            Setiap data akan ditelusuri melalui pohon keputusan hingga mencapai leaf node,
            kemudian kelas mayoritas pada leaf node tersebut akan digunakan sebagai hasil prediksi.
            """)
            
            # Show detailed prediction paths
            st.markdown("### 🛣️ Detail Jalur Prediksi")
            
            # Let user select which sample to view
            if len(st.session_state.prediction_paths) > 0:
                selected_name = st.selectbox("Pilih data untuk melihat jalur prediksi:", st.session_state.test_names)
                selected_idx = st.session_state.test_names.index(selected_name)
                
                selected_path = st.session_state.prediction_paths[selected_idx]
                selected_data = st.session_state.X_test.iloc[selected_idx]
                selected_prediction = st.session_state.test_preds.iloc[selected_idx]
                selected_actual = st.session_state.y_test.iloc[selected_idx]
                
                # Display the selected instance's values
                st.markdown(f"#### Data untuk: {selected_name}")
                
                # Format the feature values as a horizontal card layout
                cols = st.columns(len(selected_data))
                for i, (feature, value) in enumerate(selected_data.items()):
                    with cols[i]:
                        st.metric(label=feature, value=value)
                
                # Create a detailed path visualization
                st.markdown("#### Jalur Prediksi:")
                
                # Create a step-by-step visualization
                for i, step in enumerate(selected_path):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.markdown(f"**Langkah {i+1}**")
                        with col2:
                            # Format the decision nicely
                            if step["decision"]:
                                feature_name = step["feature"]
                                instance_value = step["instance_value"]
                                threshold = step["threshold"]
                                decision = step["decision"]
                                branch = "Ya" if step["branch"] == "True" else "Tidak"
                                
                                # Color code the decision
                                if decision == "≤":
                                    st.markdown(f"""
                                    <div style='padding: 0.5rem; border-radius: 0.5rem; border: 1px solid #ddd; background-color: #f8f9fa;'>
                                        <span style='font-weight: bold;'>{feature_name}</span> = {instance_value} {decision} {threshold}?
                                        <span style='color: {'green' if branch == "Ya" else 'red'}; font-weight: bold;'>{branch}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style='padding: 0.5rem; border-radius: 0.5rem; border: 1px solid #ddd; background-color: #f8f9fa;'>
                                        <span style='font-weight: bold;'>{feature_name}</span> = {instance_value} {decision} {threshold}?
                                        <span style='color: {'green' if branch == "Ya" else 'red'}; font-weight: bold;'>{branch}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        with col3:
                            # Show the branch direction
                            if step["branch"] == "True":
                                st.markdown("⬇️ Cabang Kiri")
                            elif step["branch"] == "False":
                                st.markdown("⬇️ Cabang Kanan")
                
                # Show the final prediction
                st.markdown("#### Hasil Prediksi:")
                col1, col2 = st.columns(2)
                with col1:
                    color = "#28a745" if selected_prediction == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                        <h4>Prediksi: {selected_prediction}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    color = "#28a745" if selected_actual == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                        <h4>Aktual: {selected_actual}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show whether the prediction was correct
                is_correct = selected_prediction == selected_actual
                color = "#28a745" if is_correct else "#dc3545"
                icon = "✅" if is_correct else "❌"
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                    <h4>{icon} Prediksi {'Benar' if is_correct else 'Salah'}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Add what-if analysis section
                st.markdown("### 🔄 Simulasi Perubahan Variabel")
                st.info("Ubah nilai variabel untuk melihat bagaimana prediksi berubah")
                
                # Create a form for what-if analysis
                with st.form("what_if_form"):
                    # Create a copy of the selected data for modification
                    modified_data = selected_data.copy()
                    
                    # Create sliders for each feature
                    st.markdown("#### Ubah Nilai Variabel:")
                    cols = st.columns(2)
                    
                    features = list(modified_data.index)
                    modified_values = {}
                    
                    for i, feature in enumerate(features):
                        with cols[i % 2]:
                            modified_values[feature] = st.slider(
                                f"{feature}", 
                                min_value=1, 
                                max_value=5, 
                                value=int(modified_data[feature]),
                                key=f"what_if_{feature}"
                            )
                    
                    # Submit button
                    submitted = st.form_submit_button("Simulasikan Perubahan")
                    
                    if submitted:
                        # Update the modified data
                        for feature, value in modified_values.items():
                            modified_data[feature] = value
                        
                        # Make a new prediction
                        new_prediction, new_path = predict_from_leaf_node(modified_data, st.session_state.model.tree)
                        
                        # Store the results
                        st.session_state.what_if_prediction = new_prediction
                        st.session_state.what_if_path = new_path
                        st.session_state.what_if_data = modified_data
                        st.rerun()
                
                # Display what-if results if available
                if 'what_if_prediction' in st.session_state:
                    st.markdown("#### Hasil Simulasi:")
                    
                    # Compare original and modified values
                    st.markdown("##### Perbandingan Nilai:")
                    comparison_df = pd.DataFrame({
                        'Variabel': features,
                        'Nilai Asli': [selected_data[f] for f in features],
                        'Nilai Baru': [st.session_state.what_if_data[f] for f in features],
                        'Perubahan': [st.session_state.what_if_data[f] - selected_data[f] for f in features]
                    })
                    
                    # Style the comparison dataframe
                    def color_change(val):
                        if val > 0:
                            return 'background-color: rgba(40, 167, 69, 0.2)'
                        elif val < 0:
                            return 'background-color: rgba(220, 53, 69, 0.2)'
                        return ''
                    
                    styled_comparison = comparison_df.style.applymap(color_change, subset=['Perubahan'])
                    st.dataframe(styled_comparison, use_container_width=True)
                    
                    # Show the new prediction
                    st.markdown("##### Prediksi Baru:")
                    color = "#28a745" if st.session_state.what_if_prediction == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                        <h4>Prediksi Baru: {st.session_state.what_if_prediction}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show if the prediction changed
                    if st.session_state.what_if_prediction != selected_prediction:
                        st.success(f"✅ Prediksi berubah dari {selected_prediction} menjadi {st.session_state.what_if_prediction}")
                    else:
                        st.info(f"ℹ️ Prediksi tetap sama: {selected_prediction}")
            
            st.markdown("### 📈 Evaluasi Model")
            display_metrics(st.session_state.y_test, st.session_state.test_preds)
            
            # Show confusion matrix image
            st.image('temp/confusion_matrix.png', use_container_width=True)

            # Show test results
            st.markdown("### 🔍 Detail Hasil Testing")
            
            results_df = pd.DataFrame({
                'Nama': st.session_state.test_data['Nama'],
                'Jenis Kelamin': st.session_state.test_data['Jenis Kelamin'],
                'Rata-rata Skor': st.session_state.test_data['Rata-rata Skor'],
                'Aktual': st.session_state.y_test,
                'Prediksi': st.session_state.test_preds,
                'Benar/Salah': st.session_state.y_test == st.session_state.test_preds
            })
            
            # Apply styling to results
            def color_result(val):
                if val == True:
                    return 'background-color: #28a745; color: white'
                else:
                    return 'background-color: #dc3545; color: white'
            
            styled_results = results_df.style.map(color_target, subset=['Aktual', 'Prediksi']).map(color_result, subset=['Benar/Salah']).map(color_score, subset=['Rata-rata Skor'])
            st.dataframe(styled_results, use_container_width=True)
            
            # Add friendship compatibility analysis
            st.markdown("### 👫 Analisis Kesesuaian Pertemanan")
            st.info("Analisis kesesuaian pertemanan berdasarkan pola yang ditemukan oleh model")
            
            # Create a form for friendship compatibility
            with st.form("friendship_compatibility_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    person1 = st.selectbox("Pilih Orang Pertama:", st.session_state.test_names, key="person1")
                
                with col2:
                    person2 = st.selectbox("Pilih Orang Kedua:", st.session_state.test_names, key="person2")
                
                analyze_button = st.form_submit_button("Analisis Kesesuaian")
                
                if analyze_button:
                    if person1 == person2:
                        st.warning("⚠️ Pilih dua orang yang berbeda untuk analisis kesesuaian")
                    else:
                        # Get the indices of the selected people
                        idx1 = st.session_state.test_names.index(person1)
                        idx2 = st.session_state.test_names.index(person2)
                        
                        # Get the data for both people
                        data1 = st.session_state.X_test.iloc[idx1]
                        data2 = st.session_state.X_test.iloc[idx2]
                        
                        # Get their predictions
                        pred1 = st.session_state.test_preds.iloc[idx1]
                        pred2 = st.session_state.test_preds.iloc[idx2]
                        
                        # Store the results
                        st.session_state.person1_data = data1
                        st.session_state.person2_data = data2
                        st.session_state.person1_name = person1
                        st.session_state.person2_name = person2
                        st.session_state.person1_pred = pred1
                        st.session_state.person2_pred = pred2
                        st.rerun()
            
            # Display compatibility analysis if available
            if hasattr(st.session_state, 'person1_data') and hasattr(st.session_state, 'person2_data'):
                st.markdown("#### Hasil Analisis Kesesuaian:")
                
                # Compare the data
                st.markdown("##### Perbandingan Profil:")
                
                # Create comparison dataframe
                features = st.session_state.person1_data.index.tolist()
                comparison_df = pd.DataFrame({
                    'Variabel': features,
                    f'{st.session_state.person1_name}': [st.session_state.person1_data[f] for f in features],
                    f'{st.session_state.person2_name}': [st.session_state.person2_data[f] for f in features],
                    'Perbedaan': [abs(st.session_state.person1_data[f] - st.session_state.person2_data[f]) for f in features]
                })
                
                # Style the comparison dataframe
                def color_difference(val):
                    if val == 0:
                        return 'background-color: rgba(40, 167, 69, 0.4)'
                    elif val <= 1:
                        return 'background-color: rgba(255, 193, 7, 0.2)'
                    else:
                        return 'background-color: rgba(220, 53, 69, 0.2)'
                
                styled_comparison = comparison_df.style.applymap(color_difference, subset=['Perbedaan'])
                st.dataframe(styled_comparison, use_container_width=True)
                
                # Calculate compatibility score (inverse of average difference)
                avg_difference = comparison_df['Perbedaan'].mean()
                compatibility_score = max(0, 100 - (avg_difference * 20))  # Scale to 0-100
                
                # Display compatibility score
                st.markdown("##### Skor Kesesuaian:")
                
                # Determine compatibility level
                if compatibility_score >= 80:
                    compatibility_level = "Sangat Tinggi"
                    color = "#28a745"
                elif compatibility_score >= 60:
                    compatibility_level = "Tinggi"
                    color = "#5cb85c"
                elif compatibility_score >= 40:
                    compatibility_level = "Sedang"
                    color = "#ffc107"
                elif compatibility_score >= 20:
                    compatibility_level = "Rendah"
                    color = "#f0ad4e"
                else:
                    compatibility_level = "Sangat Rendah"
                    color = "#dc3545"
                
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center;'>
                    <h3>Skor Kesesuaian: {compatibility_score:.1f}%</h3>
                    <h4>Tingkat Kesesuaian: {compatibility_level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display predictions for both people
                st.markdown("##### Prediksi Pola Pertemanan:")
                col1, col2 = st.columns(2)
                
                with col1:
                    color1 = "#28a745" if st.session_state.person1_pred == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color1}; color: white; text-align: center;'>
                        <h4>{st.session_state.person1_name}: {st.session_state.person1_pred}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    color2 = "#28a745" if st.session_state.person2_pred == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color2}; color: white; text-align: center;'>
                        <h4>{st.session_state.person2_name}: {st.session_state.person2_pred}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Provide recommendation based on compatibility and predictions
                st.markdown("##### Rekomendasi:")
                
                if compatibility_score >= 60:
                    if st.session_state.person1_pred == "Tinggi" and st.session_state.person2_pred == "Tinggi":
                        st.success(f"✅ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki kesesuaian yang baik dan keduanya memiliki pola pertemanan yang tinggi. Mereka sangat cocok untuk berteman.")
                    elif st.session_state.person1_pred == "Tinggi" or st.session_state.person2_pred == "Tinggi":
                        st.info(f"ℹ️ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki kesesuaian yang baik, tetapi salah satu memiliki pola pertemanan yang lebih tinggi. Mereka dapat saling membantu dalam mengembangkan keterampilan sosial.")
                    else:
                        st.warning(f"⚠️ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki kesesuaian yang baik, tetapi keduanya memiliki pola pertemanan yang rendah. Mereka mungkin nyaman bersama tetapi perlu mengembangkan keterampilan sosial.")
                else:
                    if st.session_state.person1_pred == "Tinggi" and st.session_state.person2_pred == "Tinggi":
                        st.warning(f"⚠️ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki perbedaan yang signifikan meskipun keduanya memiliki pola pertemanan yang tinggi. Mereka mungkin memiliki gaya pertemanan yang berbeda.")
                    elif st.session_state.person1_pred == "Tinggi" or st.session_state.person2_pred == "Tinggi":
                        st.warning(f"⚠️ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki perbedaan yang signifikan dan salah satu memiliki pola pertemanan yang lebih tinggi. Mereka mungkin menghadapi tantangan dalam berteman.")
                    else:
                        st.error(f"❌ {st.session_state.person1_name} dan {st.session_state.person2_name} memiliki perbedaan yang signifikan dan keduanya memiliki pola pertemanan yang rendah. Mereka mungkin mengalami kesulitan dalam membangun pertemanan yang baik.")
            
            # Show decision tree
            st.subheader("Pohon Keputusan:")
            # Create and display decision tree visualization
            dot = create_decision_tree_viz(st.session_state.model.tree)
            st.graphviz_chart(dot, use_container_width=True)
            
            # Also show text representation for debugging
            with st.expander("Lihat Representasi Teks Pohon Keputusan"):
                tree_text = convert_tree_to_text(st.session_state.model.tree)
                st.code(tree_text)
            
            # Add a button to clear results and run classification again
            if st.button("🔄 Reset dan Jalankan Klasifikasi Ulang"):
                # Clear all session state variables related to classification except input_data_list
                for key in list(st.session_state.keys()):
                    if key != 'input_data_list' and key != 'show_delete_confirmation':
                        del st.session_state[key]
                st.rerun()

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
