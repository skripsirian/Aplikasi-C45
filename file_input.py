import streamlit as st
import pandas as pd
from c45 import DecisionTreeC45
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Fungsi Plot Confusion Matrix
def plot_confusion_matrix(cm, classes=['Rendah', 'Tinggi']):
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages for annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.zeros_like(cm, dtype=float)
    
    # Calculate percentages, handling division by zero
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_sum[i] > 0:
                cm_perc[i, j] = (cm[i, j] / cm_sum[i]) * 100
            else:
                cm_perc[i, j] = 0
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n{cm_perc[i, j]:.1f}%'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes,
                annot_kws={'va': 'center'})
    plt.title('Confusion Matrix\n(Count and Percentage)', pad=20)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

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

        if 'Nama' not in data.columns or 'Jenis Kelamin' not in data.columns or 'target' not in data.columns:
            st.error("Dataset harus memiliki kolom 'Nama', 'Jenis Kelamin', dan 'target'.")
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
        
        # Initialize session state for test sizes if not exists
        if 'test_sizes' not in st.session_state:
            st.session_state.test_sizes = {}
            # Generate complementary random test sizes (total 100%)
            first_size = round(random.uniform(0.3, 0.7), 2)  # Between 30% and 70%
            second_size = round(1 - first_size, 2)  # Complement to make 100%
            
            # Randomly assign sizes to classes
            class_names = list(class_counts.index)
            if random.random() < 0.5:  # 50% chance to swap sizes
                st.session_state.test_sizes[class_names[0]] = first_size
                st.session_state.test_sizes[class_names[1]] = second_size
            else:
                st.session_state.test_sizes[class_names[0]] = second_size
                st.session_state.test_sizes[class_names[1]] = first_size
        
        # Display current test sizes
        st.write("### 🎲 Test Size per Kelas")
        total_percentage = 0
        for class_name in class_counts.index:
            percentage = st.session_state.test_sizes[class_name] * 100
            total_percentage += percentage
            st.info(f"Kelas '{class_name}': {percentage:.0f}% dari total data kelas")
        st.success(f"Total persentase: {total_percentage:.0f}%")
        
        # Add randomize button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎲 Acak Ulang"):
                # Generate new complementary random test sizes
                first_size = round(random.uniform(0.3, 0.7), 2)
                second_size = round(1 - first_size, 2)
                
                # Randomly assign sizes to classes
                class_names = list(class_counts.index)
                if random.random() < 0.5:
                    st.session_state.test_sizes[class_names[0]] = first_size
                    st.session_state.test_sizes[class_names[1]] = second_size
                else:
                    st.session_state.test_sizes[class_names[0]] = second_size
                    st.session_state.test_sizes[class_names[1]] = first_size
                st.rerun()
        
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
            
            # Split each class separately with its own test size
            train_indices = []
            test_indices = []
            
            for class_name in class_counts.index:
                class_indices = y[y == class_name].index
                class_size = len(class_indices)
                
                if class_size > 0:
                    # Calculate number of test samples for this class
                    n_test = int(class_size * st.session_state.test_sizes[class_name])
                    if n_test == 0:  # Ensure at least 1 test sample if possible
                        n_test = 1 if class_size > 1 else 0
                    
                    # Randomly shuffle indices
                    shuffled_indices = list(class_indices)
                    random.shuffle(shuffled_indices)
                    
                    # Split indices
                    test_indices.extend(shuffled_indices[:n_test])
                    train_indices.extend(shuffled_indices[n_test:])
            
            # Create train and test sets
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
            
        else:
            try:
                # Split each class separately
                train_indices = []
                test_indices = []
                
                for class_name in class_counts.index:
                    class_indices = y[y == class_name].index
                    class_size = len(class_indices)
                    
                    if class_size >= min_samples_per_class:
                        # Calculate number of test samples for this class
                        n_test = int(class_size * st.session_state.test_sizes[class_name])
                        if n_test < 1:  # Ensure at least 1 test sample
                            n_test = 1
                        
                        # Randomly shuffle indices
                        shuffled_indices = list(class_indices)
                        random.shuffle(shuffled_indices)
                        
                        # Split indices
                        test_indices.extend(shuffled_indices[:n_test])
                        train_indices.extend(shuffled_indices[n_test:])
                
                # Create train and test sets
                X_train = X.loc[train_indices]
                X_test = X.loc[test_indices]
                y_train = y.loc[train_indices]
                y_test = y.loc[test_indices]
                
                st.success("✅ Berhasil melakukan split dengan test size yang berbeda untuk setiap kelas")
            except Exception as e:
                st.warning(f"⚠️ Terjadi kesalahan saat splitting: {str(e)}")
                st.info("ℹ️ Beralih ke random split standar...")
                # Fallback to standard random split with average test size
                avg_test_size = sum(st.session_state.test_sizes.values()) / len(st.session_state.test_sizes)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=avg_test_size,
                    random_state=42
                )
        
        # Display split results with class distribution
        st.write("\n### 📈 Hasil Pembagian Data")
        st.write(f"Total data: {len(X)}")
        st.write(f"Data training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        st.write(f"Data testing: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Show class distribution in training and testing sets
        train_dist = pd.Series(y_train).value_counts()
        test_dist = pd.Series(y_test).value_counts()
        
        st.write("\n📊 Distribusi Kelas pada Data Training:")
        for class_name in sorted(train_dist.index):
            count = train_dist.get(class_name, 0)
            percentage = (count / len(y_train)) * 100
            st.write(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)")
            
        st.write("\n📊 Distribusi Kelas pada Data Testing:")
        for class_name in sorted(test_dist.index):
            count = test_dist.get(class_name, 0)
            percentage = (count / len(y_test)) * 100
            st.write(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)")

        if st.button("Jalankan Klasifikasi"):
            model = DecisionTreeC45()
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            preds = model.predict(X_test)
            metrics, report = model.calculate_metrics(y_test, preds)
            cm = model.get_confusion_matrix()

            st.success("Klasifikasi selesai!")
            
            # Create results dataframe for test data
            test_data = data.iloc[X_test.index]
            results_df = pd.DataFrame({
                'Nama': test_data['Nama'],
                'Jenis Kelamin': test_data['Jenis Kelamin'],
                'Aktual': y_test,
                'Prediksi': preds
            })

            def color_pred(val):
                if val == 'Tinggi':
                    return 'background-color: #28a745; color: white'
                elif val == 'Rendah':
                    return 'background-color: #dc3545; color: white'
                return ''

            styled_results = results_df.style.applymap(color_pred, subset=['Prediksi'])
            st.write("Hasil Prediksi (Data Testing):")
            st.dataframe(styled_results, use_container_width=True)

            st.markdown("### Evaluasi Model")
            display_metrics(report, metrics['accuracy']*100)

            plot_confusion_matrix(cm)
            st.image('temp/confusion_matrix.png')

            st.subheader("Pohon Keputusan:")
            st.code(model.tree)

if __name__ == "__main__":
    st.set_page_config(page_title="Klasifikasi C4.5 - Pola Pertemanan Siswa", layout="wide")
    st.title("📊 Aplikasi Klasifikasi C4.5 Pola Pertemanan Siswa")

    show_file_input()
