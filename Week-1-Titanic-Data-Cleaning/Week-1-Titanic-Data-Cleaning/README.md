# Week 1 – Titanic Data Cleaning Project

**Course:** Machine Learning & AI  
**Topic:** ML Fundamentals + Data Preprocessing  
**Mini Project:** Titanic Survival Prediction – Data Cleaning Project

## Project objective

This project uses the Titanic training dataset to practise loading and exploring data with Pandas, handling missing values, encoding categorical columns, visualizing age distribution, and exporting a cleaned CSV file.

## Project structure

```text
Week-1-Titanic-Data-Cleaning/
├── data/
│   ├── titanic_train.csv
│   └── titanic_cleaned.csv
├── images/
│   └── titanic_age_distribution.png
├── titanic_assignment.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Tasks completed

1. Loaded the Titanic training dataset with Pandas.
2. Inspected the dataset with `info()`, `describe()`, and missing-value counts.
3. Filled missing `Age` values with the median (28.0).
4. Filled missing `Embarked` values with the mode (`S`).
5. Removed `Cabin` because 687 of its 891 values were missing.
6. Applied `LabelEncoder` to `Sex` and `OneHotEncoder` to `Embarked`.
7. Created an age-distribution histogram using Matplotlib.
8. Saved the cleaned dataset to `data/titanic_cleaned.csv`.

## Results

- Original data: 891 rows and 12 columns.
- Missing Age values before cleaning: 177.
- Missing Embarked values before cleaning: 2.
- Missing Cabin values before cleaning: 687.
- Cleaned data: 891 rows and 15 columns.
- Missing values remaining after cleaning: 0.

## Requirements

- Python 3.9 or newer
- VS Code (recommended) or another Python IDE

## How to run in VS Code

1. Download or clone this repository and open its folder in VS Code.
2. Open the VS Code terminal (**Terminal → New Terminal**).
3. (Recommended) Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it:
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
5. Install the libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Run the project:

   ```bash
   python titanic_assignment.py
   ```

The script reads `data/titanic_train.csv`, updates `data/titanic_cleaned.csv`, and saves the graph in `images/`.

## Dataset

The included `titanic_train.csv` is the training dataset from the Kaggle Titanic: Machine Learning from Disaster competition. Dataset source: https://www.kaggle.com/competitions/titanic/data

## Note

This is a data-cleaning and preprocessing project. It does not train or evaluate a survival prediction model; the focus is the Week 1 assignment tasks.
