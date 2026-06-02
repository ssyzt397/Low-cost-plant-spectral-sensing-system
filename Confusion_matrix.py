import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings

"""
Purpose:
Compute evaluation metrics and visualize the confusion matrix for classification results.

Instructions:
Two parts need to be modified when switching models:
1. Update the file path of the test results (see bottom of script).
2. Update the model name in the confusion matrix title (see plot function).

Notes:
- The input Excel file must contain at least 3 columns:
  [ID, True Label, Predicted Label]
"""

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.size'] = 16  # Font size
plt.rcParams['font.family'] = 'Arial'  # Font type
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(true_labels, pred_labels, classes):
    """
    Plot confusion matrix using matplotlib.

    Parameters:
    - true_labels: Ground truth labels
    - pred_labels: Predicted labels
    - classes: List of class labels
    """
    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Modify model name here if needed
    ax.set_title('Confusion Matrix of KNN Model', fontsize=18, pad=20)

    ax.set_xlabel('Predicted Labels', fontsize=17, labelpad=15)
    ax.set_ylabel('True Labels', fontsize=17, labelpad=15)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels([str(c) for c in classes], fontsize=15)
    ax.set_yticklabels([str(c) for c in classes], fontsize=15)

    # Add values inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=20,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


def calculate_multiclass_metrics(file_path):
    """
    Calculate classification metrics and display confusion matrix.

    Parameters:
    - file_path: Path to Excel file containing prediction results

    Returns:
    - overall_df: DataFrame with overall metrics
    - class_metrics: DataFrame with per-class metrics
    """
    try:
        # Load prediction results
        df = pd.read_excel(file_path, header=0)

        print("First 5 rows of data:")
        print(df.head())
        print("-" * 60)

        # Extract true and predicted labels
        true_labels = df.iloc[:, 1]  # True labels (2nd column)
        pred_labels = df.iloc[:, 2]  # Predicted labels (3rd column)

        # Get class information
        unique_classes = sorted(list(set(true_labels)))

        print(f"Number of classes: {len(unique_classes)}")
        print(f"Class labels: {unique_classes}")
        print(f"Total samples: {len(true_labels)}")
        print("-" * 60)

        # Compute evaluation metrics
        acc = accuracy_score(true_labels, pred_labels)

        recall_per_class = recall_score(
            true_labels, pred_labels, average=None, zero_division=0
        )

        precision_per_class = precision_score(
            true_labels, pred_labels, average=None, zero_division=0
        )

        f1_per_class = f1_score(
            true_labels, pred_labels, average=None, zero_division=0
        )

        print("Per-class evaluation metrics:")

        class_metrics = pd.DataFrame({
            'Class': unique_classes,
            'Recall': [round(x, 4) for x in recall_per_class],
            'Precision': [round(x, 4) for x in precision_per_class],
            'F1-Score': [round(x, 4) for x in f1_per_class]
        })

        print(class_metrics.to_string(index=False))
        print("-" * 60)

        # Overall metrics
        overall_metrics = {
            'Evaluation Metric': ['Accuracy (Acc)'],
            'Value': [round(acc, 4)]
        }

        overall_df = pd.DataFrame(overall_metrics)

        print("Overall evaluation metric:")
        print("-" * 40)
        print(overall_df.to_string(index=False))
        print("-" * 40)

        # Plot confusion matrix
        plot_confusion_matrix(true_labels, pred_labels, unique_classes)

        return overall_df, class_metrics

    except FileNotFoundError:
        print(f"Error: File {file_path} not found! Please check the file path.")

    except IndexError:
        print(
            "Error: Insufficient columns in Excel file! Ensure there are at least 3 columns (ID, True Label, Predicted Label)."
        )

    except Exception as e:
        print(f"Unknown error occurred: {str(e)}")


if __name__ == "__main__":
    # Modify file path here for different models
    excel_path = "./Test result/KNN_Test set prediction results.xlsx"

    calculate_multiclass_metrics(excel_path)