# Import necessary libraries
import numpy as np
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import ShuffleSplit

# Load the data
data = pd.read_csv('C:/Users/Mohit/OneDrive/Desktop/geopathic/geopathic_stress_data.csv')

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values
data = data.dropna()

# Convert categorical data to numeric if necessary
# Assuming 'location' is a categorical feature, we will encode it
if 'location' in data.columns:
    data['location'] = data['location'].astype('category').cat.codes

# Ensure all data is numeric
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data = data.dropna()

# Feature selection
features = ['magnetic_field', 'conductivity', 'water_flow', 'heart_rate', 'blood_pressure']
X = data[features]
y = data['gs_zone']

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check if dataset size is sufficient for cross-validation
n_splits = 3 if len(data) < 5 else 5

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create ShuffleSplit cross-validator
shuffle_split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=shuffle_split)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {np.mean(cv_scores)}')

# Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()


# Save the trained model
joblib.dump(rf_model, 'rf_model.joblib')
