import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train the model and evaluate metrics per training step
def train_and_evaluate_with_metrics(df):
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize variables to track metrics
    training_sizes = range(10, len(X_train), len(X_train) // 10)
    metrics = {'Training Size': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    # Train the model incrementally
    for size in training_sizes:
        # Use a subset of the training data
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        
        # Train the model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_subset, y_train_subset)
        
        # Evaluate on the test set
        y_pred = rf_model.predict(X_test)
        metrics['Training Size'].append(size)
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['F1 Score'].append(f1_score(y_test, y_pred, average='weighted'))
    
    return metrics

# Visualize metrics
def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))
    
    # Plot individual metrics
    plt.plot(metrics['Training Size'], metrics['Accuracy'], label='Accuracy', marker='o')
    plt.plot(metrics['Training Size'], metrics['Precision'], label='Precision', marker='o')
    plt.plot(metrics['Training Size'], metrics['Recall'], label='Recall', marker='o')
    plt.plot(metrics['Training Size'], metrics['F1 Score'], label='F1 Score', marker='o')
    
    # Customize plot
    plt.title("Model Metrics vs Training Size", fontsize=16)
    plt.xlabel("Training Size", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Separate curve visualizations
    for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics['Training Size'], metrics[metric_name], label=metric_name, marker='o', color='b')
        plt.title(f"{metric_name} vs Training Size", fontsize=16)
        plt.xlabel("Training Size", fontsize=14)
        plt.ylabel(f"{metric_name} Score", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load dataset
    file_path = 'crop_data.csv'  # Update with your CSV file path
    df = load_data(file_path)
    
    # Train and evaluate with metrics
    metrics = train_and_evaluate_with_metrics(df)
    
    # Plot the metrics
    plot_metrics(metrics)
