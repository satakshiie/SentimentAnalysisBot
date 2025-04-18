import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("updated_cleaned_symptom_dataset.csv")

# Handle missing values by filling NaNs with 0
df = df.fillna(0)

# Check that no missing values remain
print(df.isnull().sum())  # Should show 0 for all columns

# Define input and output
X = df["text"]
label_cols = ["anxiety", "depression", "stress", "anger", "frustration"]
y = df[label_cols]

# Combine X and y
df_combined = pd.concat([X, y], axis=1)

# Separate each class
stress_df = df_combined[df_combined["stress"] == 1]
no_stress_df = df_combined[df_combined["stress"] == 0]

# Oversample the underrepresented classes
anger_df = df_combined[df_combined["anger"] == 1]
anger_oversample = resample(anger_df, replace=True, n_samples=len(stress_df), random_state=42)

# Similarly oversample other underrepresented classes if necessary (for example "frustration")

# Concatenate back the oversampled data
df_balanced = pd.concat([stress_df, anger_oversample, no_stress_df])  # Add other oversampled classes here if necessary

# Shuffle the combined and oversampled dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the balanced data into features (X) and labels (y)
X_balanced = df_balanced["text"]
y_balanced = df_balanced[label_cols]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),  # Text to vector conversion
    ("clf", OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))  # One-vs-rest classification
])

# Cross-validation on the balanced data
scores = cross_val_score(pipeline, X_balanced, y_balanced, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")

# Train model
pipeline.fit(X_train, y_train)

# Save model and label names
joblib.dump(pipeline, "symptom_checker_model.pkl")
joblib.dump(label_cols, "symptom_labels.pkl")

print("✅ Model and label names saved!")