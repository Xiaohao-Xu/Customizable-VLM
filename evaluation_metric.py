import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from collections import defaultdict
import json

pred_file = './output/answer_internvl2_8b.json'
output_filename = "./output/answer_internvl2_8b.txt"

with open(pred_file, 'r') as f:
    data = json.load(f)

# Overall
predictions = [item['correctness'] for item in data]
labels = [item['label'] for item in data]
avg_acc = accuracy_score(labels, predictions)
avg_auroc = roc_auc_score(labels, predictions)
avg_aupr = average_precision_score(labels, predictions)


# Initialize dictionaries to store ground truth and predictions for each object_type
y_true = defaultdict(list)
y_pred = defaultdict(list)

# Process the data
for item in data:
    object_type = item["object_type"]
    label = item["label"]
    correctness = item["correctness"]
    y_true[object_type].append(label)
    y_pred[object_type].append(correctness)

# Function to compute metrics for each object_type
def compute_metrics(y_true, y_pred):
    results = {}
    for object_type in y_true:
        true_labels = np.array(y_true[object_type])
        predicted_labels = np.array(y_pred[object_type])
        # Accuracy
        acc = accuracy_score(true_labels, predicted_labels)
        # AUROC
        try:
            auroc = roc_auc_score(true_labels, predicted_labels)
        except ValueError:
            auroc = np.nan  # If AUROC cannot be calculated (e.g., all labels are the same)
        # AUPR
        try:
            aupr = average_precision_score(true_labels, predicted_labels)
        except ValueError:
            aupr = np.nan  # If AUPR cannot be calculated (e.g., all labels are the same)
        results[object_type] = {"ACC": acc, "AUROC": auroc, "AUPR": aupr}
    return results
# Compute the metrics
metrics = compute_metrics(y_true, y_pred)

with open(output_filename, "w") as f:
    for object_type, metric_values in metrics.items():
        f.write(f"Metrics for {object_type}:\n")
        f.write(f"  ACC: {metric_values['ACC']:.3f}\n")
        f.write(f"  AUROC: {metric_values['AUROC']:.3f}\n")
        f.write(f"  AUPR: {metric_values['AUPR']:.3f}\n")
        f.write("\n")
    # Write average metrics
    f.write(f"Average Metrics:\n")
    f.write(f"  Average ACC: {avg_acc:.3f}\n")
    f.write(f"  Average AUROC: {avg_auroc:.3f}\n")
    f.write(f"  Average AUPR: {avg_aupr:.3f}\n")
    
    

print(f"Metrics saved to {output_filename}")