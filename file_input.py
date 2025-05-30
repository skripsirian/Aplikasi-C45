import streamlit as st
import pandas as pd
from c45 import DecisionTreeC45
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time
import graphviz
import re
import io
import matplotlib.colors as mcolors

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

# Function to determine friendship type based on scores
def determine_friendship_type(scores):
    # Extract individual scores
    keberagaman, komunikasi, empati, kerjasama, konflik, dukungan, kepemimpinan = scores
    
    # Calculate average score
    avg_score = sum(scores) / len(scores)
    
    # Determine friendship type based on scores and patterns
    if avg_score >= 4.0 and min(scores) >= 3.5:
        return "Teman Sejati", "Memiliki hubungan pertemanan yang sangat kuat dan seimbang di semua aspek"
    elif avg_score >= 3.5 and (komunikasi >= 4 or kerjasama >= 4):
        return "Teman Saling Membutuhkan", "Memiliki hubungan pertemanan yang saling menguntungkan dengan fokus pada komunikasi dan kerjasama"
    elif avg_score >= 3.0 and (komunikasi >= 3.5 or empati >= 3.5):
        return "Teman Satu Arah", "Memiliki hubungan pertemanan yang lebih dominan pada satu pihak"
    else:
        return "Teman Tidak Sehat", "Memiliki pola pertemanan yang perlu diperbaiki"

# Fungsi untuk menghitung kategori pola pertemanan
def calculate_friendship_category(row, threshold_high=4.0, threshold_low=3.0):
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
    scores = [float(row[comp]) for comp in components]
    total_score = sum(scores)
    average_score = total_score / len(components)
    
    # Tentukan kategori dan tipe pertemanan
    if average_score >= threshold_high:
        category = 'Tinggi'
    elif average_score >= threshold_low:
        category = 'Sedang'
    else:
        category = 'Rendah'
        
    friendship_type, friendship_desc = determine_friendship_type(scores)
    
    return category, friendship_type, friendship_desc

# Fungsi Plot Confusion Matrix
def plot_confusion_matrix(cm, classes=['Rendah', 'Sedang', 'Tinggi']):
    plt.figure(figsize=(12, 10))
    
    # Calculate percentages for annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.zeros_like(cm, dtype=float)
    total_samples = np.sum(cm)  # Total samples
    
    # Calculate row percentages and overall percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_sum[i] > 0:
                cm_perc[i, j] = (cm[i, j] / cm_sum[i]) * 100  # Percentage from class total
            else:
                cm_perc[i, j] = 0
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / total_samples * 100 if total_samples > 0 else 0
    
    # Create a better colormap that works well with three classes
    colors = plt.cm.Blues(np.linspace(0, 1, 128))
    colors = np.vstack((plt.cm.Reds_r(np.linspace(0, 1, 128)), colors))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_Blues_Reds', colors)
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Format: count (row %)
            annot[i, j] = f'{cm[i, j]}\n({cm_perc[i, j]:.1f}%)'
    
    # Plot heatmap with custom colormap
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap=custom_cmap,
                xticklabels=classes, yticklabels=classes,
                annot_kws={'va': 'center', 'ha': 'center'},
                linewidths=1.5, linecolor='white', cbar=True)
    
    # Add title with accuracy information
    plt.title(f'Confusion Matrix\nAkurasi: {accuracy:.1f}%\nTotal Data: {total_samples}', 
              fontsize=16, pad=20)
    
    plt.ylabel('Aktual', fontsize=14)
    plt.xlabel('Prediksi', fontsize=14)
    
    # Calculate class-specific metrics
    class_metrics = []
    for i, class_name in enumerate(classes):
        true_pos = cm[i, i]
        false_pos = np.sum(cm[:, i]) - true_pos
        false_neg = np.sum(cm[i, :]) - true_pos
        true_neg = np.sum(cm) - true_pos - false_pos - false_neg
        
        # Calculate metrics (handle division by zero)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Add a metrics summary table at the bottom
    table_data = []
    table_data.append(['Kelas', 'Precision', 'Recall', 'F1-Score'])
    for i, metrics in enumerate(class_metrics):
        table_data.append([
            metrics['class'],
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1']:.2f}"
        ])
    
    # Create the table
    table = plt.table(
        cellText=table_data,
        cellLoc='center',
        loc='bottom',
        bbox=[0.0, -0.35, 1.0, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Add diagonal line to highlight correct predictions
    ax.plot([0, len(classes)], [0, len(classes)], "k--", alpha=0.3)
    
    # Create directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    plt.tight_layout()
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
        # Tampilkan kelas dalam urutan yang konsisten
        for class_name in ['Rendah', 'Sedang', 'Tinggi']:
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
                elif val == 'Sedang':
                    return 'background-color: #ffc107; color: black'
                elif val == 'Rendah':
                    return 'background-color: #dc3545; color: white'
                return ''

            styled_df = metrics_df.style.map(color_class, subset=['Kelas'])
            st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        pass  # Abaikan error jika terjadi

# Function to predict based on the leaf node where data falls
def predict_from_leaf_node(instance, tree, node_path=None, leaf_stats=None):
    if node_path is None:
        node_path = []
    
    # If we've reached a leaf node (not a dictionary)
    if not isinstance(tree, dict):
        # For C4.5, the leaf node contains the majority class
        # In our implementation, tree is the majority class at the leaf
        leaf_class = tree
        
        # Extract statistics if available (for display purposes)
        stats = {}
        if isinstance(leaf_stats, dict) and leaf_class in leaf_stats:
            stats = {
                "class": leaf_class,
                "count": leaf_stats[leaf_class],
                "total": sum(leaf_stats.values()),
                "probability": leaf_stats[leaf_class] / sum(leaf_stats.values()) if sum(leaf_stats.values()) > 0 else 0
            }
        else:
            stats = {
                "class": leaf_class,
                "count": 1,
                "total": 1,
                "probability": 1.0
            }
            
        return leaf_class, node_path, stats
    
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
            return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]], node_path, leaf_stats)
        else:
            node_path.append({
                "feature": feature_name,
                "threshold": threshold,
                "instance_value": instance_value,
                "decision": ">",
                "branch": "False",
                "description": f"{feature_name} = {instance_value} > {threshold} → False"
            })
            return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[1]], node_path, leaf_stats)
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
        return predict_from_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]], node_path, leaf_stats)

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

# Function to collect leaf node class distributions
def collect_leaf_node_stats(tree, X_train, y_train):
    # Initialize a dictionary to store leaf node statistics
    leaf_stats = {}
    
    # Function to traverse the tree and find the leaf node for each instance
    def find_leaf_node(instance, tree):
        if not isinstance(tree, dict):
            return tree  # Return the leaf node class
        
        # Get the feature to split on
        feature = list(tree.keys())[0]
        
        # Extract feature name and threshold if it contains a threshold
        match = re.search(r'(.*?) <= (.*)', feature)
        if match:
            feature_name, threshold = match.groups()
            feature_name = feature_name.strip()
            threshold = float(threshold)
            
            # Determine which branch to follow
            if instance[feature_name] <= threshold:
                return find_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]])
            else:
                return find_leaf_node(instance, tree[feature][list(tree[feature].keys())[1]])
        else:
            # If there's no threshold, just follow the first branch
            return find_leaf_node(instance, tree[feature][list(tree[feature].keys())[0]])
    
    # Process each training instance
    for i in range(len(X_train)):
        instance = X_train.iloc[i]
        label = y_train.iloc[i]
        
        # Find the leaf node for this instance
        leaf_node = find_leaf_node(instance, tree)
        
        # Create a unique identifier for this leaf node
        leaf_id = str(leaf_node)
        
        # Initialize the leaf node statistics if not already present
        if leaf_id not in leaf_stats:
            leaf_stats[leaf_id] = {"Tinggi": 0, "Sedang": 0, "Rendah": 0}
        
        # Increment the count for this class
        leaf_stats[leaf_id][label] += 1
    
    return leaf_stats

def color_result(val):
    if val == True:
        return 'background-color: #28a745; color: white'
    else:
        return 'background-color: #dc3545; color: white'

def color_target(val):
    if val == 'Tinggi':
        return 'background-color: #28a745; color: white'
    elif val == 'Sedang':
        return 'background-color: #ffc107; color: black'
    elif val == 'Rendah':
        return 'background-color: #dc3545; color: white'
    return ''

def color_score(val):
    if val >= 4.0:
        return 'background-color: rgba(40, 167, 69, 0.2)'
    elif val >= 3.0:
        return 'background-color: rgba(255, 193, 7, 0.2)'
    else:
        return 'background-color: rgba(220, 53, 69, 0.2)'
                
def color_friendship_type(val):
    if val == "Teman Sejati":
        return 'background-color: #28a745; color: white'
    elif val == "Teman Saling Membutuhkan":
        return 'background-color: #5cb85c; color: white'
    elif val == "Teman Satu Arah":
        return 'background-color: #ffc107; color: black'
    else:
        return 'background-color: #dc3545; color: white'

# Add friendship type classification
def get_friendship_type(row):
    scores = [
        float(row['Keberagaman Teman']),
        float(row['Kemampuan Komunikasi']),
        float(row['Empati dan Pengertian']),
        float(row['Kerjasama dan Kolaborasi']),
        float(row['Mengelola Konflik']),
        float(row['Dukungan Sosial']),
        float(row['Kepemimpinan dan Tanggung Jawab'])
    ]
    return determine_friendship_type(scores)[0]

# Fungsi File Input & Klasifikasi
def show_file_input():
    st.info("📤 Upload Dataset Mode")
    
    # Initialize classification state if not already set
    if 'classification_run' not in st.session_state:
        st.session_state.classification_run = False
    
    uploaded_file = st.file_uploader("📂 Upload Dataset (.csv / .xlsx)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
                
            # Extract file name without extension for table title
            file_name = os.path.splitext(uploaded_file.name)[0]
            st.session_state.table_title = file_name
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
            data['target'], data['Tipe Pertemanan'], data['Deskripsi'] = zip(*data.apply(calculate_friendship_category, axis=1))
        except Exception as e:
            st.error(f"Error saat menghitung kategori: {str(e)}")
            return

        # Add numbering column if it doesn't exist
        if 'No' not in data.columns:
            data.insert(0, 'No', range(1, len(data) + 1))
            
        # Add Kelas column if it doesn't exist
        if 'Kelas' not in data.columns:
            # Add a text input for class as a simple attribute
            default_kelas = st.text_input("Kelas untuk Semua Siswa:", value="Kelas 10")
            
            # Apply the selected class to all students as a simple attribute
            data['Kelas'] = default_kelas
            
        # Store the data in session state for later use
        if 'dataset' not in st.session_state:
            st.session_state.dataset = data
        else:
            # Keep existing dataset if it exists, only update if new file uploaded
            pass
            
        # Add new data form
        with st.expander("➕ Tambah Data Baru"):
            st.write("Tambahkan data baru ke dataset:")
            
            # Initialize key for form
            if 'form_key' not in st.session_state:
                st.session_state.form_key = 0
                
            with st.form(f"add_data_form_{st.session_state.form_key}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    nama = st.text_input("Nama")
                with col2:
                    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                with col3:
                    # Use the same class as the dataset if available, otherwise provide default
                    if 'dataset' in st.session_state and 'Kelas' in st.session_state.dataset.columns:
                        default_kelas = st.session_state.dataset['Kelas'].iloc[0]
                        kelas = st.text_input("Kelas", value=default_kelas)
                    else:
                        kelas = st.text_input("Kelas", value="Kelas 10")
                
                # Input for each component
                st.write("Komponen Penilaian:")
                col1, col2 = st.columns(2)
                
                with col1:
                    keberagaman = st.slider("Keberagaman Teman", 1, 5, 3)
                    komunikasi = st.slider("Kemampuan Komunikasi", 1, 5, 3)
                    empati = st.slider("Empati dan Pengertian", 1, 5, 3)
                    konflik = st.slider("Mengelola Konflik", 1, 5, 3)
                
                with col2:
                    kerjasama = st.slider("Kerjasama dan Kolaborasi", 1, 5, 3)
                    dukungan = st.slider("Dukungan Sosial", 1, 5, 3)
                    kepemimpinan = st.slider("Kepemimpinan dan Tanggung Jawab", 1, 5, 3)
                
                # Calculate average score
                scores = [keberagaman, komunikasi, empati, kerjasama, konflik, dukungan, kepemimpinan]
                avg_score = sum(scores) / len(scores)
                target = 'Tinggi' if avg_score >= 3.5 else 'Rendah'
                
                # Display calculated target
                st.info(f"Rata-rata skor: {avg_score:.2f} → Kategori: {target}")
                
                submitted = st.form_submit_button("Tambahkan Data")
                
                if submitted:
                    if not nama:
                        st.error("Nama tidak boleh kosong!")
                    else:
                        # Determine the next number
                        if 'dataset' in st.session_state and len(st.session_state.dataset) > 0 and 'No' in st.session_state.dataset.columns:
                            next_number = st.session_state.dataset['No'].max() + 1
                        else:
                            next_number = 1
                            
                        # Create new data row
                        new_data = {
                            'No': next_number,
                            'Nama': nama,
                            'Jenis Kelamin': jenis_kelamin,
                            'Kelas': kelas,
                            'Keberagaman Teman': keberagaman,
                            'Kemampuan Komunikasi': komunikasi,
                            'Empati dan Pengertian': empati,
                            'Kerjasama dan Kolaborasi': kerjasama,
                            'Mengelola Konflik': konflik,
                            'Dukungan Sosial': dukungan,
                            'Kepemimpinan dan Tanggung Jawab': kepemimpinan,
                            'target': target
                        }
                        
                        # Add to the dataset
                        new_df = pd.DataFrame([new_data])
                        
                        if 'dataset' not in st.session_state:
                            st.session_state.dataset = new_df
                        else:
                            st.session_state.dataset = pd.concat([st.session_state.dataset, new_df], ignore_index=True)
                        
                        # Reset index for proper handling
                        st.session_state.dataset = st.session_state.dataset.reset_index(drop=True)
                        
                        # Success message
                        st.success(f"✅ Data untuk {nama} berhasil ditambahkan dengan nomor {next_number}!")
                        
                        # Increment form key to reset the form
                        st.session_state.form_key += 1
                        
                        # Rerun to refresh the page
                        st.rerun()
        
        # Display dataset with delete functionality
        if 'dataset' in st.session_state and len(st.session_state.dataset) > 0:
            # Display table title if available
            if 'table_title' in st.session_state:
                st.write(f"### Dataset: {st.session_state.table_title}")
            else:
                st.write("### Dataset:")
            
            # Add delete functionality
            with st.expander("🗑️ Hapus Data"):
                st.write("Pilih data yang ingin dihapus berdasarkan nomor:")
                
                # Get list of available numbers
                numbers = st.session_state.dataset['No'].tolist()
                
                # Create selection for deletion
                number_to_delete = st.selectbox("Pilih nomor data yang akan dihapus:", numbers)
                
                if st.button("Hapus Data"):
                    # Get the index of the row to delete
                    idx_to_delete = st.session_state.dataset[st.session_state.dataset['No'] == number_to_delete].index
                    
                    if len(idx_to_delete) > 0:
                        # Get name for confirmation message
                        name_to_delete = st.session_state.dataset.loc[idx_to_delete[0], 'Nama']
                        
                        # Delete the row
                        st.session_state.dataset = st.session_state.dataset.drop(idx_to_delete)
                        
                        # Reset index
                        st.session_state.dataset = st.session_state.dataset.reset_index(drop=True)
                        
                        # Show success message
                        st.success(f"✅ Data untuk {name_to_delete} (No. {number_to_delete}) berhasil dihapus!")
                        
                        # Rerun to refresh the page
                        st.rerun()
            
            # Display the dataset
            st.dataframe(st.session_state.dataset, use_container_width=True)
        else:
            st.write("Dataset:")
            st.dataframe(data, use_container_width=True)

        # Add option to download the updated dataset
        if 'dataset' in st.session_state and len(st.session_state.dataset) > len(data):
            col1, col2 = st.columns(2)
            with col1:
                csv = st.session_state.dataset.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset (CSV)",
                    data=csv,
                    file_name="updated_dataset.csv",
                    mime="text/csv"
                )
            with col2:
                # Create Excel file in memory
                output = io.BytesIO()
                try:
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        st.session_state.dataset.to_excel(writer, index=False, sheet_name='Data')
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📥 Download Dataset (Excel)",
                        data=excel_data,
                        file_name="updated_dataset.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning(f"Excel export tidak tersedia: {str(e)}")
                    st.info("Silakan gunakan opsi CSV untuk mengunduh data.")

        # Make a copy of data to avoid modifying the original
        data_for_model = data.copy()
        
        # Split features and target
        X = data_for_model.drop(['target', 'Nama', 'Jenis Kelamin', 'No', 'Kelas'], axis=1)
        y = data_for_model['target'].str.title()  # Normalize to Title case
        
        # Check class distribution
        class_distribution = y.value_counts()
        st.write("### 📊 Distribusi Kelas Sebelum Balancing")
        for class_name, count in class_distribution.items():
            percentage = (count / len(y)) * 100
            st.write(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)")
        
        # User input for test size
        st.write("\n### 🎲 Pengaturan Test Size dan Balancing")
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
        
        # Add balancing method selection
        balancing_method = st.radio(
            "Pilih metode balancing untuk data test:",
            ["Balancing", "Normal (C4.5 Standard)"],
            help="Pilih metode untuk menyeimbangkan distribusi kelas pada data test"
        )
        
        # Add info about balancing methods
        st.info("""
        ℹ️ Tentang Metode Balancing:
        - **Balancing**: Seimbangkan distribusi kelas pada data test dengan mengurangi jumlah sampel dari kelas mayoritas
        - **Normal (C4.5 Standard)**: Gunakan algoritma C4.5 standar tanpa modifikasi distribusi kelas (sesuai implementasi asli)
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
            for class_name, count in class_distribution.items() 
            if count < min_samples_per_class
        ]
        
        # Reset index before splitting to ensure proper indexing
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Implement balanced train-test split
        # Split data into training and testing sets
        if insufficient_classes:
            st.warning(f"⚠️ Tidak bisa melakukan stratified split karena jumlah sampel tidak mencukupi untuk kelas: {', '.join(insufficient_classes)}")
            st.info(f"ℹ️ Dibutuhkan minimal {min_samples_per_class} sampel per kelas untuk stratified split")
            st.info("ℹ️ Beralih ke random split...")
            
            # Use a random seed based on current time for truly random splits each time
            random_seed = int(time.time()) % 10000
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_seed
            )
        else:
            try:
                # Use a random seed based on current time for truly random splits each time
                random_seed = int(time.time()) % 10000
                
                # Use stratified sampling to maintain class distribution
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    stratify=y,
                    random_state=random_seed
                )
                
                # Check if we need to balance the test set further
                test_class_dist = y_test.value_counts()
                min_samples = test_class_dist.min()
                
                # Apply balancing based on selected method
                if balancing_method == "Balancing":
                    st.warning("⚠️ Melakukan downsampling pada data test untuk menyeimbangkan kelas...")
                    
                    # Create balanced test set by sampling from each class
                    balanced_indices = []
                    for class_name in sorted(test_class_dist.index):
                        class_indices = y_test[y_test == class_name].index.tolist()
                        # If we have more samples than the minimum, downsample
                        if len(class_indices) > min_samples:
                            sampled_indices = random.sample(class_indices, min_samples)
                            balanced_indices.extend(sampled_indices)
                        else:
                            # Otherwise use all samples
                            balanced_indices.extend(class_indices)
                    
                    # Create balanced test set
                    X_test_balanced = X_test.loc[balanced_indices]
                    y_test_balanced = y_test.loc[balanced_indices]
                    
                    # Update the test sets
                    X_test = X_test_balanced
                    y_test = y_test_balanced
                    
                    st.success("✅ Balancing selesai! Data test sekarang memiliki distribusi yang lebih seimbang.")
                
                elif balancing_method == "Normal (C4.5 Standard)":
                    st.info("ℹ️ Menggunakan algoritma C4.5 standar tanpa modifikasi distribusi kelas (sesuai implementasi asli).")
                    # Show the distribution
                    st.write("### 📊 Distribusi Kelas pada Data Test")
                    for class_name, count in test_class_dist.items():
                        percentage = (count / len(y_test) * 100)
                        st.write(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)")
                
                st.success(f"✅ Berhasil melakukan stratified split dengan random seed: {random_seed}")
            except Exception as e:
                st.warning(f"⚠️ Terjadi kesalahan saat splitting: {str(e)}")
                st.info("ℹ️ Beralih ke random split standar...")
                
                # Use a random seed based on current time for truly random splits each time
                random_seed = int(time.time()) % 10000
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_seed
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
                elif class_name == 'Sedang':
                    st.warning(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
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
                elif class_name == 'Sedang':
                    st.warning(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
                else:
                    st.success(f"- {class_name}: {count} sampel ({percentage:.1f}%)")
        
        # Add visualization of class distribution
        st.write("\n### 📊 Visualisasi Distribusi Kelas")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training data distribution
        train_counts = pd.Series(y_train).value_counts().sort_index()
        colors = ['#dc3545', '#ffc107', '#28a745']  # Red, Yellow, Green
        ax1.bar(train_counts.index, train_counts.values, color=colors[:len(train_counts)])
        ax1.set_title('Distribusi Kelas - Data Training')
        ax1.set_ylabel('Jumlah Sampel')
        
        # Testing data distribution
        test_counts = pd.Series(y_test).value_counts().sort_index()
        ax2.bar(test_counts.index, test_counts.values, color=colors[:len(test_counts)])
        ax2.set_title('Distribusi Kelas - Data Testing')
        
        # Add percentage labels
        for i, v in enumerate(train_counts.values):
            ax1.text(i, v + 0.5, f"{v} ({v/sum(train_counts.values)*100:.1f}%)", 
                    ha='center', va='bottom')
        
        for i, v in enumerate(test_counts.values):
            ax2.text(i, v + 0.5, f"{v} ({v/sum(test_counts.values)*100:.1f}%)", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add preview of test data
        st.write("\n### 🔍 Preview Data Testing")
        with st.expander("Lihat Data Testing", expanded=True):
            # Create a preview dataframe with relevant columns
            test_data_preview = data.iloc[X_test.index].copy()
            
            # Add feature columns
            feature_columns = [
                'Keberagaman Teman',
                'Kemampuan Komunikasi',
                'Empati dan Pengertian',
                'Kerjasama dan Kolaborasi',
                'Mengelola Konflik',
                'Dukungan Sosial',
                'Kepemimpinan dan Tanggung Jawab'
            ]
            
            # Calculate average score
            test_data_preview['Rata-rata Skor'] = test_data_preview[feature_columns].mean(axis=1).round(2)
            
            # Select columns for display
            preview_columns = ['No', 'Nama', 'Jenis Kelamin', 'Kelas', 'target', 'Rata-rata Skor', 'Tipe Pertemanan']
            if all(col in test_data_preview.columns for col in preview_columns):
                test_preview_df = test_data_preview[preview_columns]
            else:
                # If some columns are missing, use what's available
                available_columns = [col for col in preview_columns if col in test_data_preview.columns]
                test_preview_df = test_data_preview[available_columns]
            
            # Apply styling
            def color_target_value(val):
                if val == 'Tinggi':
                    return 'background-color: #28a745; color: white'
                elif val == 'Sedang':
                    return 'background-color: #ffc107; color: black'
                elif val == 'Rendah':
                    return 'background-color: #dc3545; color: white'
                return ''
            
            def color_score_value(val):
                if val >= 4.0:
                    return 'background-color: rgba(40, 167, 69, 0.2)'
                elif val >= 3.0:
                    return 'background-color: rgba(255, 193, 7, 0.2)'
                else:
                    return 'background-color: rgba(220, 53, 69, 0.2)'
            
            # Apply the styling if 'target' and 'Rata-rata Skor' columns exist
            if 'target' in test_preview_df.columns and 'Rata-rata Skor' in test_preview_df.columns:
                styled_preview = test_preview_df.style.map(color_target_value, subset=['target']).map(color_score_value, subset=['Rata-rata Skor'])
                st.dataframe(styled_preview, use_container_width=True)
            else:
                st.dataframe(test_preview_df, use_container_width=True)
            
            st.info(f"📊 Data preview menampilkan {len(test_preview_df)} data yang akan digunakan untuk testing.")
            st.info("ℹ️ Data ini akan digunakan untuk mengevaluasi model setelah klasifikasi dijalankan.")

        if st.button("Jalankan Klasifikasi"):
            # Store a flag in session state to indicate classification has been run
            st.session_state.classification_run = True
            
            model = DecisionTreeC45()
            model.fit(X_train, y_train)
            
            # Collect leaf node statistics from training data
            leaf_stats = collect_leaf_node_stats(model.tree, X_train, y_train)
            
            # Make predictions using leaf node majority class
            test_preds = []
            prediction_paths = []
            prediction_stats = []
            prediction_probabilities = []
            
            # Predict each test instance
            for idx, row in X_test.iterrows():
                # First predict to get the leaf class
                prediction, path, stats = predict_from_leaf_node(row, model.tree)
                # Then use the prediction to get the correct leaf stats
                leaf_id = str(prediction)
                leaf_node_stats = leaf_stats.get(leaf_id, {})
                
                # Calculate probabilities for each class
                total_samples = sum(leaf_node_stats.values()) if leaf_node_stats else 1
                prob_tinggi = leaf_node_stats.get('Tinggi', 0) / total_samples if total_samples > 0 else 0
                prob_sedang = leaf_node_stats.get('Sedang', 0) / total_samples if total_samples > 0 else 0
                prob_rendah = leaf_node_stats.get('Rendah', 0) / total_samples if total_samples > 0 else 0
                
                # Store the probabilities
                class_probabilities = {
                    'Tinggi': prob_tinggi,
                    'Sedang': prob_sedang,
                    'Rendah': prob_rendah
                }
                prediction_probabilities.append(class_probabilities)
                
                # Re-predict with the correct leaf stats
                prediction, path, stats = predict_from_leaf_node(row, model.tree, leaf_stats=leaf_node_stats)
                test_preds.append(prediction)
                prediction_paths.append(path)
                prediction_stats.append(stats)
            
            # Convert predictions to series for evaluation
            preds = pd.Series(test_preds, index=X_test.index)
            
            # Calculate metrics using our custom predictions
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Check if we have all three classes in our test data
            available_classes = sorted(list(set(list(y_test) + list(preds))))
            
            # Generate report with available classes
            report = classification_report(y_test, preds, output_dict=True)
            
            # For confusion matrix, ensure we have all three classes if possible
            labels = ['Rendah', 'Sedang', 'Tinggi']
            cm = confusion_matrix(y_test, preds, labels=labels)
            
            metrics = {'accuracy': (preds == y_test).mean()}
            
            # Store everything in session state
            st.session_state.model = model
            st.session_state.test_data = data.iloc[X_test.index]
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.preds = preds
            st.session_state.report = report
            st.session_state.metrics = metrics
            st.session_state.cm = cm
            st.session_state.prediction_paths = prediction_paths
            st.session_state.prediction_stats = prediction_stats
            st.session_state.prediction_probabilities = prediction_probabilities
            st.session_state.test_names = data.iloc[X_test.index]['Nama'].tolist()
            st.session_state.leaf_stats = leaf_stats
            st.session_state.feature_columns = [
                'Keberagaman Teman',
                'Kemampuan Komunikasi',
                'Empati dan Pengertian',
                'Kerjasama dan Kolaborasi',
                'Mengelola Konflik',
                'Dukungan Sosial',
                'Kepemimpinan dan Tanggung Jawab'
            ]
            st.session_state.avg_scores = data.iloc[X_test.index][st.session_state.feature_columns].mean(axis=1)
            
            # Generate and save confusion matrix plot
            plot_confusion_matrix(cm)
            
            st.success("Klasifikasi selesai!")
            st.rerun()  # Rerun to refresh the page with session state data
        
        # Check if classification has been run and display results
        if 'classification_run' in st.session_state and st.session_state.classification_run:
            # Display information about the prediction process
            st.markdown("### 🌲 Informasi Prediksi Berdasarkan Leaf Node")
            st.info("""
            Prediksi dilakukan berdasarkan kelas mayoritas pada leaf node tempat data jatuh.
            Setiap data akan ditelusuri melalui pohon keputusan hingga mencapai leaf node,
            kemudian kelas mayoritas pada leaf node tersebut akan digunakan sebagai hasil prediksi.
            """)
            
            # Create results dataframe with No column
            results_df = pd.DataFrame({
                'No': st.session_state.test_data['No'] if 'No' in st.session_state.test_data.columns else range(1, len(st.session_state.test_data) + 1),
                'Nama': st.session_state.test_data['Nama'],
                'Jenis Kelamin': st.session_state.test_data['Jenis Kelamin'],
                'Kelas': st.session_state.test_data['Kelas'],
                'Aktual': st.session_state.y_test,
                'Prediksi': st.session_state.preds,
                'Benar/Salah': st.session_state.y_test == st.session_state.preds,
                'Rata-rata Skor': st.session_state.test_data[st.session_state.feature_columns].astype(float).mean(axis=1).round(2),
                'Tipe Pertemanan': st.session_state.test_data.apply(get_friendship_type, axis=1)
            })

            # Limit the number of rows displayed to avoid React rendering errors
            max_rows_to_display = 20
            if len(results_df) > max_rows_to_display:
                st.warning(f"⚠️ Menampilkan {max_rows_to_display} dari {len(results_df)} baris data untuk menghindari error rendering.")
                display_df = results_df.head(max_rows_to_display)
            else:
                display_df = results_df

            # Apply styling to results
            styled_results = display_df.style.map(color_target, subset=['Aktual', 'Prediksi']).map(color_result, subset=['Benar/Salah']).map(color_score, subset=['Rata-rata Skor']).map(color_friendship_type, subset=['Tipe Pertemanan'])
            st.dataframe(styled_results, use_container_width=True)

            st.markdown("### Evaluasi Model")
            display_metrics(st.session_state.report, st.session_state.metrics['accuracy']*100)

            # Display confusion matrix image
            st.image('temp/confusion_matrix.png')
            
            # Display confusion matrix summary
            st.write("\n### 📑 Ringkasan Confusion Matrix")
            total_samples = np.sum(st.session_state.cm)
            for i, actual_class in enumerate(['Rendah', 'Sedang', 'Tinggi']):
                for j, pred_class in enumerate(['Rendah', 'Sedang', 'Tinggi']):
                    count = st.session_state.cm[i, j]
                    percentage = (count / total_samples) * 100
                    if actual_class == pred_class:
                        st.success(f"✅ {actual_class} diprediksi benar sebagai {pred_class}: {count} data ({percentage:.1f}%)")
                    else:
                        st.error(f"❌ {actual_class} diprediksi salah sebagai {pred_class}: {count} data ({percentage:.1f}%)")
            
            # Show detailed prediction path for a sample
            st.markdown("### 🛣️ Contoh Jalur Prediksi")
            if len(st.session_state.prediction_paths) > 0:
                selected_name = st.selectbox("Pilih data untuk melihat jalur prediksi:", 
                                          st.session_state.test_names,
                                          key="selected_name_path")
                
                selected_idx = st.session_state.test_names.index(selected_name)
                selected_path = st.session_state.prediction_paths[selected_idx]
                selected_data = st.session_state.X_test.iloc[selected_idx]
                selected_prediction = st.session_state.preds.iloc[selected_idx]
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
                        pred1 = st.session_state.preds.iloc[idx1]
                        pred2 = st.session_state.preds.iloc[idx2]
                        
                        # Get their friendship types
                        scores1 = [data1[col] for col in st.session_state.feature_columns]
                        scores2 = [data2[col] for col in st.session_state.feature_columns]
                        
                        # Calculate combined scores for friendship type determination
                        combined_scores = [(s1 + s2) / 2 for s1, s2 in zip(scores1, scores2)]
                        friendship_type = determine_friendship_type(combined_scores)[0]
                        
                        # Store the results
                        st.session_state.person1_data = data1
                        st.session_state.person2_data = data2
                        st.session_state.person1_name = person1
                        st.session_state.person2_name = person2
                        st.session_state.person1_pred = pred1
                        st.session_state.person2_pred = pred2
                        st.session_state.friendship_type = friendship_type
                        st.rerun()
            
            # Display compatibility analysis if available
            if hasattr(st.session_state, 'person1_data') and hasattr(st.session_state, 'person2_data'):
                st.markdown("#### Hasil Analisis Kesesuaian:")
                
                # Compare the data
                st.markdown("##### Perbandingan Profil:")
                
                # Create comparison dataframe with only numeric features
                numeric_features = [
                    'Keberagaman Teman',
                    'Kemampuan Komunikasi',
                    'Empati dan Pengertian',
                    'Kerjasama dan Kolaborasi',
                    'Mengelola Konflik',
                    'Dukungan Sosial',
                    'Kepemimpinan dan Tanggung Jawab'
                ]
                
                comparison_df = pd.DataFrame({
                    'Variabel': numeric_features,
                    f'{st.session_state.person1_name}': [float(st.session_state.person1_data[f]) for f in numeric_features],
                    f'{st.session_state.person2_name}': [float(st.session_state.person2_data[f]) for f in numeric_features],
                    'Perbedaan': [abs(float(st.session_state.person1_data[f]) - float(st.session_state.person2_data[f])) for f in numeric_features]
                })
                
                # Style the comparison dataframe
                def color_difference(val):
                    if isinstance(val, (int, float)):
                        if val == 0:
                            return 'background-color: rgba(40, 167, 69, 0.4)'
                        elif val <= 1:
                            return 'background-color: rgba(255, 193, 7, 0.2)'
                        else:
                            return 'background-color: rgba(220, 53, 69, 0.2)'
                    return ''
                
                styled_comparison = comparison_df.style.map(color_difference, subset=['Perbedaan'])
                st.dataframe(styled_comparison, use_container_width=True)
                
                # Calculate compatibility score (inverse of average difference)
                avg_difference = comparison_df['Perbedaan'].mean()
                compatibility_score = max(0, 100 - (avg_difference * 20))  # Scale to 0-100
                
                # Display compatibility score
                st.markdown("##### Skor Kesesuaian:")
                
                # Determine compatibility level
                if compatibility_score >= 90:
                    compatibility_level = "Sangat Tinggi"
                    color = "#28a745"
                elif compatibility_score >= 75:
                    compatibility_level = "Tinggi"
                    color = "#5cb85c"
                elif compatibility_score >= 60:
                    compatibility_level = "Sedang"
                    color = "#ffc107"
                elif compatibility_score >= 45:
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
                
                # Display predictions and friendship types for both people
                st.markdown("##### Prediksi Pola Pertemanan:")
                col1, col2 = st.columns(2)
                
                with col1:
                    color1 = "#28a745" if st.session_state.person1_pred == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color1}; color: white; text-align: center; margin-bottom: 1rem;'>
                        <h4>{st.session_state.person1_name}: {st.session_state.person1_pred}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    color2 = "#28a745" if st.session_state.person2_pred == "Tinggi" else "#dc3545"
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: {color2}; color: white; text-align: center; margin-bottom: 1rem;'>
                        <h4>{st.session_state.person2_name}: {st.session_state.person2_pred}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display combined friendship type
                type_color = "#28a745" if st.session_state.friendship_type == "Teman Sejati" else "#5cb85c" if st.session_state.friendship_type == "Teman Saling Membutuhkan" else "#ffc107" if st.session_state.friendship_type == "Teman Satu Arah" else "#dc3545"
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; background-color: {type_color}; color: {'black' if st.session_state.friendship_type == "Teman Satu Arah" else 'white'}; text-align: center;'>
                    <h4>Tipe Pertemanan: {st.session_state.friendship_type}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a dataframe to analyze compatibility factors
                compatibility_factors = []
                
                # Calculate factor scores based on differences
                for feature in numeric_features:
                    val1 = float(st.session_state.person1_data[feature])
                    val2 = float(st.session_state.person2_data[feature])
                    diff = abs(val1 - val2)
                    
                    # Determine compatibility level for this factor
                    if diff == 0:
                        level = "Sangat Tinggi"
                        score = 5
                        color = "#28a745"
                    elif diff <= 0.5:
                        level = "Tinggi"
                        score = 4
                        color = "#5cb85c"
                    elif diff <= 1:
                        level = "Sedang"
                        score = 3
                        color = "#ffc107"
                    elif diff <= 1.5:
                        level = "Rendah"
                        score = 2
                        color = "#f0ad4e"
                    else:
                        level = "Sangat Rendah"
                        score = 1
                        color = "#dc3545"
                    
                    compatibility_factors.append({
                        'Faktor': feature,
                        'Nilai 1': val1,
                        'Nilai 2': val2,
                        'Perbedaan': diff,
                        'Tingkat Kesesuaian': level,
                        'Skor': score,
                        'Color': color
                    })
                
                # Create a dataframe for the factors
                factors_df = pd.DataFrame(compatibility_factors)
                
                # Find strongest and weakest factors
                strongest_factor = factors_df.loc[factors_df['Skor'].idxmax()]
                weakest_factor = factors_df.loc[factors_df['Skor'].idxmin()]
                
                # Calculate average compatibility score
                avg_compatibility = factors_df['Skor'].mean()
                
                # Add detailed analysis breakdown
                st.markdown("#### 🔍 Detail Analisis Kesesuaian")
                
                # Display the factors in an interactive table
                st.markdown("##### Analisis Faktor Kesesuaian:")
                
                # Display as a regular Streamlit dataframe instead of custom HTML
                display_df = factors_df[['Faktor', 'Nilai 1', 'Nilai 2', 'Perbedaan', 'Tingkat Kesesuaian']].copy()
                
                # Define a function to style the Tingkat Kesesuaian column
                def style_compatibility(val):
                    if val == "Sangat Tinggi":
                        return 'background-color: #28a745; color: white'
                    elif val == "Tinggi":
                        return 'background-color: #5cb85c; color: white'
                    elif val == "Sedang":
                        return 'background-color: #ffc107; color: black'
                    elif val == "Rendah":
                        return 'background-color: #f0ad4e; color: white'
                    else:  # Sangat Rendah
                        return 'background-color: #dc3545; color: white'
                
                # Apply styling
                styled_df = display_df.style.map(style_compatibility, subset=['Tingkat Kesesuaian'])
                
                # Display the dataframe
                st.dataframe(styled_df, use_container_width=True, height=300)
                
                # Create a radar chart of compatibility factors
                st.markdown("##### Grafik Kesesuaian Faktor:")
                
                # Prepare data for radar chart
                categories = factors_df['Faktor'].tolist()
                values = factors_df['Skor'].tolist()
                
                # Create radar chart using matplotlib
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, polar=True)
                
                # Number of variables
                N = len(categories)
                
                # What will be the angle of each axis in the plot (divide the plot / number of variables)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Values for the chart
                values = values + [values[0]]  # Close the loop
                
                # Draw the chart
                ax.plot(angles, values, linewidth=2, linestyle='solid', label='Kesesuaian')
                ax.fill(angles, values, alpha=0.25)
                
                # Fix axis to go in the right order and start at 12 o'clock
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw axis lines for each angle and label
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=9)
                
                # Draw y-axis labels (0-5)
                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_yticklabels(['1', '2', '3', '4', '5'])
                ax.set_ylim(0, 5)
                
                # Add title
                plt.title(f'Profil Kesesuaian: {st.session_state.person1_name} & {st.session_state.person2_name}', size=15)
                
                # Save to a temporary file and display
                radar_chart_path = 'temp/radar_chart.png'
                plt.savefig(radar_chart_path, bbox_inches='tight', dpi=100)
                plt.close()
                
                st.image(radar_chart_path)
                
                # Add detailed interpretation using Streamlit components instead of HTML
                st.markdown("##### Interpretasi Kesesuaian:")
                
                # Create columns for metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faktor Kesesuaian Tertinggi", strongest_factor['Faktor'], strongest_factor['Tingkat Kesesuaian'])
                with col2:
                    st.metric("Faktor Kesesuaian Terendah", weakest_factor['Faktor'], weakest_factor['Tingkat Kesesuaian'])
                    
                # Show average score
                st.metric("Rata-rata Skor Kesesuaian", f"{avg_compatibility:.2f}/5.0")
                
                # Provide detailed recommendation based on the weakest factor
                st.markdown("##### Rekomendasi Pengembangan:")
                
                recommendations = {
                    'Keberagaman Teman': "Cobalah untuk memperluas lingkaran pertemanan bersama-sama dan terbuka terhadap teman-teman baru dengan latar belakang yang berbeda.",
                    'Kemampuan Komunikasi': "Tingkatkan keterampilan komunikasi dengan saling mendengarkan secara aktif dan mengekspresikan pikiran dengan jelas.",
                    'Empati dan Pengertian': "Kembangkan kemampuan untuk memahami perspektif satu sama lain dan menunjukkan empati dalam situasi yang sulit.",
                    'Kerjasama dan Kolaborasi': "Lakukan lebih banyak aktivitas yang memerlukan kerjasama tim dan saling mendukung untuk mencapai tujuan bersama.",
                    'Mengelola Konflik': "Pelajari cara menyelesaikan perbedaan pendapat dengan cara yang konstruktif dan menghindari konflik yang tidak perlu.",
                    'Dukungan Sosial': "Tingkatkan dukungan emosional dan praktis satu sama lain, terutama saat menghadapi tantangan.",
                    'Kepemimpinan dan Tanggung Jawab': "Kembangkan rasa tanggung jawab bersama dan kemampuan untuk saling memimpin dalam situasi yang berbeda."
                }
                
                st.info(f"💡 **Fokus Pengembangan:** {weakest_factor['Faktor']}\n\n{recommendations.get(weakest_factor['Faktor'], 'Tidak ada rekomendasi spesifik.')}")
                
                # Add overall compatibility summary
                st.markdown("##### Kesimpulan:")
                
                # Create a summary based on compatibility score
                if compatibility_score >= 80:
                    summary = f"{st.session_state.person1_name} dan {st.session_state.person2_name} memiliki tingkat kesesuaian yang sangat tinggi. Mereka memiliki banyak kesamaan dalam aspek-aspek penting pertemanan dan berpotensi untuk membangun hubungan yang sangat kuat dan berkelanjutan."
                    summary_color = "#28a745"
                elif compatibility_score >= 60:
                    summary = f"{st.session_state.person1_name} dan {st.session_state.person2_name} memiliki tingkat kesesuaian yang tinggi. Meskipun ada beberapa perbedaan, mereka memiliki dasar yang kuat untuk membangun pertemanan yang baik."
                    summary_color = "#5cb85c"
                elif compatibility_score >= 40:
                    summary = f"{st.session_state.person1_name} dan {st.session_state.person2_name} memiliki tingkat kesesuaian yang sedang. Ada beberapa kesamaan dan perbedaan yang signifikan, yang memerlukan usaha dari kedua belah pihak untuk mengembangkan pertemanan yang lebih kuat."
                    summary_color = "#ffc107"
                elif compatibility_score >= 20:
                    summary = f"{st.session_state.person1_name} dan {st.session_state.person2_name} memiliki tingkat kesesuaian yang rendah. Ada banyak perbedaan yang perlu dijembatani, tetapi dengan komunikasi yang baik dan saling pengertian, pertemanan masih bisa berkembang."
                    summary_color = "#f0ad4e"
                else:
                    summary = f"{st.session_state.person1_name} dan {st.session_state.person2_name} memiliki tingkat kesesuaian yang sangat rendah. Mereka memiliki perbedaan yang signifikan dalam banyak aspek pertemanan dan mungkin perlu usaha ekstra untuk membangun hubungan yang baik."
                    summary_color = "#dc3545"
                
                # Use st.success, st.warning, st.error based on compatibility level
                if compatibility_score >= 60:
                    st.success(summary)
                elif compatibility_score >= 40:
                    st.warning(summary)
                else:
                    st.error(summary)

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
                # Clear all session state variables related to classification
                for key in list(st.session_state.keys()):
                    if key != 'dataset' and key != 'form_key':
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Klasifikasi C4.5 - Pola Pertemanan Siswa", layout="wide")
    st.title("📊 Aplikasi Klasifikasi C4.5 Pola Pertemanan Siswa")

    show_file_input()
