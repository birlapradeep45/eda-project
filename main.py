import pandas as pd

# Load dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

# First 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# Column names
print("\n--- Column Names ---")
print(df.columns)

# Data types
print("\n--- Data Types ---")
print(df.dtypes)

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())
# Handle missing values

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has many missing values
df = df.drop("Cabin", axis=1)

# Check missing values again
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())
# Save cleaned dataset
df.to_csv("data/cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")
# Basic statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())
# Survival Count Visualization

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))

sns.countplot(x="Survived", data=df)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()

# Save graph
plt.savefig("outputs/survival_count.png")

plt.show()
# Survival by Gender

plt.figure(figsize=(6, 4))

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()

# Save graph
plt.savefig("outputs/survival_by_gender.png")

plt.show()
# Survival by Passenger Class

plt.figure(figsize=(7, 4))

sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()

# Save graph
plt.savefig("outputs/survival_by_class.png")

plt.show()
# Age Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Age"], bins=20, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

# Save graph
plt.savefig("outputs/age_distribution.png")

plt.show()
# Correlation Heatmap

plt.figure(figsize=(10, 7))

correlation = df.select_dtypes(include="number").corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

# Save graph
plt.savefig("outputs/correlation_heatmap.png")

plt.show()
# Survival Rate by Passenger Class

survival_rate = df.groupby("Pclass")["Survived"].mean() * 100

print("\n--- Survival Rate by Passenger Class ---")
print(survival_rate)

plt.figure(figsize=(7, 4))

sns.barplot(
    x=survival_rate.index,
    y=survival_rate.values
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.tight_layout()

# Save graph
plt.savefig("outputs/survival_rate_by_class.png")

plt.show()
# Fare Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Fare"], bins=30, kde=True)

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

# Save graph
plt.savefig("outputs/fare_distribution.png")

plt.show()
# Final EDA Summary

print("\n==============================")
print("       EDA PROJECT SUMMARY")
print("==============================")

print("\nTotal Passengers:", len(df))

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nAverage Age:", round(df["Age"].mean(), 2))

print("\nAverage Fare:", round(df["Fare"].mean(), 2))

print("\nPassengers by Gender:")
print(df["Sex"].value_counts())

print("\nPassengers by Class:")
print(df["Pclass"].value_counts())

print("\nSurvival Rate by Gender (%):")
print(df.groupby("Sex")["Survived"].mean().mul(100).round(2))

print("\nSurvival Rate by Class (%):")
print(df.groupby("Pclass")["Survived"].mean().mul(100).round(2))