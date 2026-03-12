# OIBSIP_DataAnalytics--task4-L2
This project analyzes Google Play Store data to understand app market trends. It involves cleaning the dataset, exploring app categories, analyzing ratings, size, popularity, and pricing patterns, and visualizing insights to better understand the Android app ecosystem.
# Unveiling the Android App Market: Analyzing Google Play Store Data

## Objective
The objective of this project is to analyze Google Play Store data to understand app market dynamics, trends, and user preferences using data analysis and visualization techniques.

## Dataset
Dataset Link: https://www.kaggle.com/datasets/utshabkumarghosh/android-app-market-on-google-play

The dataset contains information about various Android applications including category, ratings, installs, size, price, and user reviews.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- VS Code

## Steps Performed
1. Loaded the Google Play Store dataset.
2. Cleaned the dataset by correcting data types and handling missing values.
3. Explored different app categories to understand their distribution.
4. Analyzed important metrics such as ratings, installs, app size, and pricing.
5. Examined user reviews to understand general sentiment.
6. Created visualizations to identify patterns and trends in the app market.

## Code (Example)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("googleplaystore.csv")

sns.countplot(x="Category", data=data)
plt.xticks(rotation=90)
plt.show()
Key Insights

Some categories have significantly more apps than others.

Ratings and installs indicate app popularity.

Pricing patterns vary across different app categories.

Outcome

The project provides insights into the Android app market and demonstrates how data analysis and visualization can reveal trends in app popularity, pricing, and user preferences.

Author

Ayesha Asna
