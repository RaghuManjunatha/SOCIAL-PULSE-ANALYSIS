import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Social Pulse- Sentiment Analysis.csv")

# Display the first few rows of the dataset
print("Initial dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values)

# Check for outliers
outliers = df.describe().loc[['min', 'max']]
print("\nOutliers:")
print(outliers)

# Define a function to categorize seasons based on month
def categorize_season(month):
    if month in [12, 1, 2]:  # Winter: December, January, February
        return 'Winter'
    elif month in [3, 4, 5]:  # Spring: March, April, May
        return 'Spring'
    elif month in [6, 7, 8]:  # Summer: June, July, August
        return 'Summer'
    else:  # Fall: September, October, November
        return 'Fall'

# Apply the function to create the new "Season" column
df['Season'] = df['Month'].apply(categorize_season)

# Calculate Engagement Rate
df['Engagement Rate'] = (df['Retweets'] + df['Likes']) / 2  # Average of retweets and likes


# Convert 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create 'Weekday' column
df['Weekday'] = df['Timestamp'].dt.day_name()

# Print out some sample rows to see the 'Weekday' column
print(df[['Timestamp', 'Weekday']].head(10))


# Remove leading and trailing spaces from platform names
df['Platform'] = df['Platform'].str.strip()

# Check unique platform names after cleaning
print(df['Platform'].unique())

# Check for extra spaces in the 'Final Sentiments' column
df['Final Sentiments'] = df['Final Sentiments'].str.strip()

# Ensure that only 'Positive', 'Negative', and 'Neutral' are present in the 'Final Sentiments' column
valid_sentiments = ['Positive', 'Negative', 'Neutral']
df['Final Sentiments'] = df['Final Sentiments'].apply(lambda x: x if x in valid_sentiments else 'Neutral')

print("Unique values after processing:")
print(df['Final Sentiments'].unique())

# Display summary statistics
summary_stats = df.describe(include='all')
print("Summary Statistics:")
print(summary_stats)

# Display data types and non-null values
info = df.info()
print("\nData Types and Non-null Values:")
print(info)

# Display the updated dataframe
print(df.head())


# Save the preprocessed dataset with the new column in the specified folder
df.to_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Preprocessed_dataset.csv", index=False)


