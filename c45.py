import numpy as np
import pandas as pd
from math import log2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

class DecisionTreeC45:
    def __init__(self):
        self.tree = None
        self.confusion_matrix = None
        self.metrics = None
        self.classification_report = None
        self.most_common_class = None

    def entropy(self, target_col):
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = -np.sum([(counts[i]/np.sum(counts)) * log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    def info_gain(self, data, feature, target_name):
        total_entropy = self.entropy(data[target_name])
        vals, counts = np.unique(data[feature], return_counts=True)

        weighted_entropy = np.sum([
            (counts[i]/np.sum(counts)) * self.entropy(data.where(data[feature]==vals[i]).dropna()[target_name])
            for i in range(len(vals))
        ])

        info_gain = total_entropy - weighted_entropy
        return info_gain

    def best_attribute(self, data, attributes, target_name):
        gains = [self.info_gain(data, attr, target_name) for attr in attributes]
        return attributes[np.argmax(gains)]

    def build_tree(self, data, attributes, target_name):
        labels = np.unique(data[target_name])

        if len(labels) == 1:
            return labels[0]
        if len(attributes) == 0:
            return data[target_name].mode()[0]

        best_attr = self.best_attribute(data, attributes, target_name)
        tree = {best_attr: {}}

        for val in np.unique(data[best_attr]):
            sub_data = data.where(data[best_attr] == val).dropna()
            subtree = self.build_tree(
                sub_data,
                [attr for attr in attributes if attr != best_attr],
                target_name
            )
            tree[best_attr][val] = subtree

        return tree

    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        
        self.most_common_class = y.mode()[0]
        
        self.tree = self.build_tree(data, X.columns.tolist(), 'target')

        predictions = self.predict(X)
        self.calculate_metrics(y, predictions)

        return self

    def predict_instance(self, query, tree):
        if not isinstance(tree, dict):
            return tree

        attr = next(iter(tree))
        if query[attr] in tree[attr]:
            return self.predict_instance(query, tree[attr][query[attr]])
        else:
            return self.most_common_class

    def predict(self, X):
        predictions = X.apply(lambda row: self.predict_instance(row, self.tree), axis=1)
        return predictions

    def calculate_metrics(self, y_true, y_pred):
        # Ensure both classes are present in labels
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        labels = ['Rendah', 'Tinggi'] if len(unique_classes) == 2 else unique_classes
        
        # Calculate confusion matrix with fixed labels
        self.confusion_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        try:
            # Ensure both classes are included in classification report
            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            report = {"error": str(e)}

        self.classification_report = report

        # Calculate accuracy manually to ensure correctness
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        self.metrics = {'accuracy': accuracy}
        return self.metrics, report

    def get_confusion_matrix(self):
        if self.confusion_matrix is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        return self.confusion_matrix

    def get_metrics(self):
        if self.metrics is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        return self.metrics

    def get_classification_report(self):
        if self.classification_report is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")

        report_df = pd.DataFrame(self.classification_report).transpose()
        if 'support' in report_df.columns:
            numeric_cols = ['precision', 'recall', 'f1-score']
            report_df[numeric_cols] = report_df[numeric_cols].round(4)
        return report_df

    def print_metrics(self):
        if self.metrics is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")

        print("\nModel Evaluation Metrics:")
        print("-" * 30)
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")

        print("\nClassification Report:")
        print("-" * 30)
        print(self.get_classification_report())

        print("\nConfusion Matrix:")
        print("-" * 30)
        print(pd.DataFrame(
            self.confusion_matrix,
            index=[f'Actual {c}' for c in ['Rendah', 'Tinggi']],
            columns=[f'Predicted {c}' for c in ['Rendah', 'Tinggi']]
        ))
