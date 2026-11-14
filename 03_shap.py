
# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import shap
import matplotlib.pyplot as plt

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Oversample training data only
X_train, y_train = data_loader.oversample(X_train, y_train)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# %% Fit black-box model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Stroke", "Stroke"],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Stroke", "Stroke"]
)

disp.plot(cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)

start_index = 1
end_index = 2

patient = X_test.iloc[start_index:end_index]

# Calculate SHAP values for one patient
shap_values = explainer.shap_values(patient)

# Select class 1 (Stroke)
# Newer SHAP versions return (samples, features, classes)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values[:, :, 1]

print("SHAP values shape:", shap_values_class1.shape)

# %% Visualize local predictions - Task 3
prediction = rf.predict(patient)[0]
probability = rf.predict_proba(patient)[0, 1]

print(f"The RF predicted: {prediction}")
print(f"Predicted stroke probability: {probability:.2%}")

# Generate static Force Plot
shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[0],
    patient.iloc[0],
    matplotlib=True,
    show=False,
    figsize=(16, 4)
)

# Save the current SHAP figure
plt.savefig(
    "shap_force_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# %% Visualize global features - Task 4
shap_values_global = explainer.shap_values(X_test)

# Select class 1 (Stroke)
if isinstance(shap_values_global, list):
    shap_values_global_class1 = shap_values_global[1]
else:
    shap_values_global_class1 = shap_values_global[:, :, 1]

# Generate global SHAP Summary Plot
shap.summary_plot(
    shap_values_global_class1,
    X_test,
    show=False
)

plt.savefig(
    "shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
