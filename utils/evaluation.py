import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def make_classification_report(true_labels, pred_labels, class_names):
    # Print the standard text report
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # Compute overall metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
        "macro_precision": precision_score(true_labels, pred_labels, average="macro", zero_division=0),
        "macro_recall": recall_score(true_labels, pred_labels, average="macro", zero_division=0),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
    }

    # Print a nice summary
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        print(f"{metric:>20}: {value:.4f}")

def make_confusion_matrix(true_labels, pred_labels, class_names):
    m = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 8))
    sns.heatmap(m, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
