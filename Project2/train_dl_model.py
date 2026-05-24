import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("dados_grade_invertida.csv")

# Target
y = data["GradeClass"].values

# Features
X = data.drop(["GradeClass", "GPA", "StudentID"], axis=1)

# Columns definition
categorical_cols = ["Gender", "Ethnicity"]
binary_cols = ["Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]
numerical_cols = ["Age", "ParentalEducation", "StudyTimeWeekly", "Absences"]

# Preprocessing: OneHotEncode categorical and Scale numerical
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numerical_cols),
        ("bin", "passthrough", binary_cols)
    ]
)

# Apply preprocessing
X_processed = preprocess.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Deep Learning Model
class StudentPerformanceNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StudentPerformanceNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
num_classes = len(np.unique(y))

# Compute class weights to handle imbalanced classes
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

model = StudentPerformanceNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
print("\n" + "="*50)
print(f"DEEP LEARNING MODEL EVALUATION")
print("="*50)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Deep Learning Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save plot instead of showing directly to avoid blocking script
plt.savefig("dl_confusion_matrix.png")
print("\nConfusion matrix plot saved as 'dl_confusion_matrix.png'")

import joblib
joblib.dump(preprocess, "dl_preprocessor.pkl")
print("Preprocessor saved as 'dl_preprocessor.pkl'")

# Optionally save the PyTorch model
torch.save(model.state_dict(), "dl_model.pth")
print("Model state dictionary saved as 'dl_model.pth'")
