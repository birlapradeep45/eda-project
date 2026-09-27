# Exploratory Data Analysis - Titanic Dataset

## Project Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic dataset.

The main purpose of this project is to understand the dataset, clean missing values, analyze important variables, and visualize patterns using Python.

## Objectives

- Understand the Titanic dataset
- Identify missing values
- Clean the dataset
- Perform statistical analysis
- Analyze passenger survival
- Study the relationship between gender and survival
- Study the relationship between passenger class and survival
- Analyze age and fare distributions
- Generate useful visualizations
- Identify relationships between numerical variables

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenPyXL

## Dataset

The project uses the Titanic passenger dataset.

The original dataset is stored in:

`data/Titanic-Dataset.csv`

The cleaned dataset is stored in:

`data/cleaned_titanic.csv`

## Data Cleaning

The following cleaning operations were performed:

1. Missing Age values were filled using the median age.
2. Missing Embarked values were filled using the most frequent value.
3. The Cabin column was removed because it contained many missing values.
4. The cleaned dataset was saved as a new CSV file.

## Exploratory Data Analysis

The following analyses were performed:

- Survival count
- Survival by gender
- Survival by passenger class
- Age distribution
- Fare distribution
- Correlation heatmap
- Survival rate by passenger class
- Survival rate by gender

## Project Outputs

The generated visualizations are stored in the `outputs` folder.

Important output files include:

- `survival_count.png`
- `survival_by_gender.png`
- `survival_by_class.png`
- `age_distribution.png`
- `fare_distribution.png`
- `correlation_heatmap.png`
- `survival_rate_by_class.png`

## How to Run the Project

### Step 1: Install required libraries

```bash
pip install -r requirements.txt