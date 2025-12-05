# Credit Card Approval Prediction

## Overview
Built a machine learning pipeline to predict credit card approval risk by analyzing historical applicant and credit history data, helping minimize default risk for issuers while giving applicants insight into approval likelihood before applying.

## Objective
- Help financial institutions reduce risk from issuing credit cards to unqualified applicants.
- Allow applicants to gauge approval likelihood in advance, avoiding unnecessary credit score impact from rejected applications.

## Methodology
Followed the **CRISP-DM** (Cross Industry Standard Process for Data Mining) framework end to end: business understanding, data understanding, data preparation, modeling, evaluation, and deployment planning.

## Data
- Combined two datasets: applicant background data and monthly credit status records (438,557+ rows after merging).
- Defined a target label by classifying applicants as "good" or "bad" credit risk based on payment delinquency codes (30+ days overdue, write-offs).

## Data Preparation & Feature Engineering
- Cleaned missing/null values and renamed columns for clarity.
- Converted binary Y/N fields (car, house ownership) into 1/0 integers and removed bias-prone columns such as gender.
- Binned continuous variables (family size, income) into categorical groups using quantile and equal-length cuts.
- Consolidated occupation types into three simplified categories (labor, office work, high-tech) to reduce dimensionality.
- Applied one-hot/dummy encoding to categorical features for model readiness.

## Handling Class Imbalance
- Identified a heavily skewed target distribution (~94% good vs. 6% bad credit).
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset before training.

## Modeling
- Trained and compared two classification models:
  - **Logistic Regression** — baseline accuracy of 53.3%.
  - **Random Forest Classifier** — improved accuracy of 58.2%, later refined to 73.4% through iterative feature and pipeline adjustments.
- Evaluated both models using confusion matrices and normalized accuracy scores.

## Results & Reflection
- Initial results were evaluated as insufficiently reliable for deployment.
- Iterated back through the data preparation and modeling stages per CRISP-DM to improve performance, raising Random Forest accuracy from 58% to 73%.
- Identified time constraints as the main limitation preventing further model tuning and deployment.

## Tools & Skills
Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn, Logistic Regression, Random Forest, feature engineering, exploratory data analysis, CRISP-DM methodology.