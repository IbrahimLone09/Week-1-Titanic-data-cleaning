"""Week 1: ML Fundamentals + Data Preprocessing
Mini Project: Titanic Survival Prediction – Data Cleaning Project

Run this file from the project root after installing requirements.txt.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

# 1. Load and explore the Titanic training dataset
df = pd.read_csv(DATA_DIR / "titanic_train.csv")
print("DATASET INFORMATION")
df.info()
print("\nBASIC STATISTICS")
print(df.describe())
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

# 2. Handle missing values
# Age is numeric: use the median to reduce the effect of extreme values.
df["Age"] = df["Age"].fillna(df["Age"].median())
# Embarked is categorical: fill missing values with its most frequent category.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Cabin has many missing values, so remove it for this introductory exercise.
df = df.drop(columns=["Cabin"])

# 3. Encode categorical variables
label_encoder = LabelEncoder()
df["Sex_Encoded"] = label_encoder.fit_transform(df["Sex"])
print("\nSEX ENCODING:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# OneHotEncoder creates one binary column for each Embarked category.
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:  # Support older scikit-learn versions.
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[["Embarked"]])
encoded_columns = [f"Embarked_{category}" for category in encoder.categories_[0]]
encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index).astype(int)
df = pd.concat([df, encoded_df], axis=1)

# 4. Visualize the age distribution
plt.figure(figsize=(9, 5))
plt.hist(df["Age"], bins=30, edgecolor="black")
plt.title("Titanic Passengers: Age Distribution")
plt.xlabel("Age (years)")
plt.ylabel("Number of passengers")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "titanic_age_distribution.png", dpi=160)
plt.show()

# 5. Export and verify the cleaned dataset
df.to_csv(DATA_DIR / "titanic_cleaned.csv", index=False)
print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
print(f"\nCleaned dataset saved to: {DATA_DIR / 'titanic_cleaned.csv'}")
print(f"Age histogram saved to: {IMAGE_DIR / 'titanic_age_distribution.png'}")
