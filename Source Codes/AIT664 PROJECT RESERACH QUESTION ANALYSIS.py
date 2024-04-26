import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/PROJECT UPDATE 2/Preprocessed_dataset.csv")

"""
Question 1 : How do sentiment expression vary acorss platforms and locations?
"""

# Plot bar plot for sentiment distribution by platform
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', hue='Final Sentiments', data=df, palette='coolwarm')
plt.title('Sentiment Distribution by Platform')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.show()

# Get the top 5 countries based on frequency
top_countries = df['Country'].value_counts().head(5).index.tolist()

print("Top 5 countries in the dataset:")
print(top_countries)

# Remove leading and trailing spaces from the 'Country' column
df['Country'] = df['Country'].str.strip()

# Get the top 5 countries based on frequency
top_countries = df['Country'].value_counts().head(5).index.tolist()

print("Top 5 countries in the dataset:")
print(top_countries)

# Get the top 5 countries based on frequency
top_countries = df['Country'].value_counts().head(5).index.tolist()

# Filter the dataframe for the top 5 countries
df_top_countries = df[df['Country'].isin(top_countries)]

# Plot grouped bar plot for final sentiments by top 5 countries
plt.figure(figsize=(10, 6))
sns.countplot(x='Country', hue='Final Sentiments', data=df_top_countries, palette='coolwarm')
plt.title('Final Sentiments by Top 5 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation= 0 )
plt.legend(title='Final Sentiments')
plt.show()


"""
Question 2 : Is there a temporal pattern in sentiment expression over time?

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/PROJECT UPDATE 2/preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Map sentiment categories to numerical labels
sentiment_labels = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Sentiment Label'] = df['Final Sentiments'].map(sentiment_labels)

# Extract different time components
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Season'] = df['Timestamp'].dt.month.apply(lambda x: 'Spring' if 3 <= x <= 5 else ('Summer' if 6 <= x <= 8 else ('Autumn' if 9 <= x <= 11 else 'Winter')))

# Group by month and calculate average sentiment
monthly_sentiment = df.groupby('Month')['Sentiment Label'].mean()

# Group by day and calculate average sentiment
daily_sentiment = df.groupby('Day')['Sentiment Label'].mean()

# Group by hour and calculate average sentiment
hourly_sentiment = df.groupby('Hour')['Sentiment Label'].mean()

# Group by season and calculate average sentiment
seasonal_sentiment = df.groupby('Season')['Sentiment Label'].mean()

# Plot temporal patterns
plt.figure(figsize=(12, 6))
monthly_sentiment.plot(marker='o')
plt.title('Average Sentiment by Month')
plt.xlabel('Month')
plt.ylabel('Average Sentiment')
plt.xticks(range(1, 13))  # Show all 12 months
plt.show()

plt.figure(figsize=(12, 6))
daily_sentiment.plot(marker='o')
plt.title('Average Sentiment by Day')
plt.xlabel('Day')
plt.ylabel('Average Sentiment')
plt.xticks(range(1, 31))  # Show all 30 days
plt.show()

plt.figure(figsize=(12, 6))
hourly_sentiment.plot(marker='o')
plt.title('Average Sentiment by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Sentiment')
plt.xticks(range(24))  # Show all 24 hours
plt.show()

plt.figure(figsize=(12, 6))
seasonal_sentiment.plot(kind='bar')
plt.title('Average Sentiment by Season')
plt.xlabel('Season')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=0)  # Rotate x-axis labels to 0 degrees
plt.show()


"""
Question 3 : How do engagement metrics correlate with sentiment across platforms
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# Assuming you have a dataset named 'dataset.csv' with columns: 'Retweets', 'Likes', 'Sentiment', 'Platform'
dataset = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/PROJECT UPDATE 2/Preprocessed_dataset.csv")

# Encode categorical sentiment values into numerical representations
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
dataset['Final Sentiments'] = dataset['Final Sentiments'].map(sentiment_mapping)

# Encode categorical platform values into numerical representations
platform_mapping = {'Twitter': 0, 'Instagram': 1, 'Facebook': 2}  # You may need to adjust this mapping based on your dataset
dataset['Platform'] = dataset['Platform'].map(platform_mapping)


# Select relevant columns
engagement_sentiment = dataset[['Retweets', 'Likes', 'Final Sentiments']]

# Perform correlation analysis
correlation_matrix = engagement_sentiment.corr()

# Visualize correlation matrix using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Engagement Metrics and Sentiment')
plt.show()

# Visualize relationship between engagement metrics and sentiment using scatter plots
sns.pairplot(engagement_sentiment, hue='Platform', markers=['o', 's', 'D'], palette='Set1')
plt.suptitle('Engagement Metrics and Sentiment across Platforms', y=1.02)  # Adjust the y parameter for vertical position
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/PROJECT UPDATE 2/preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Define numerical labels for sentiment categories
sentiment_labels = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Sentiment Label'] = df['Final Sentiments'].map(sentiment_labels)

# Plot box plots for engagement metrics vs. sentiment categories
plt.figure(figsize=(12, 6))

# Box plot for likes vs. sentiment
plt.subplot(1, 2, 1)
sns.boxplot(x='Final Sentiments', y='Likes', data=df, palette='Set1')
plt.title('Likes vs. Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Likes')

# Box plot for retweets vs. sentiment
plt.subplot(1, 2, 2)
sns.boxplot(x='Final Sentiments', y='Retweets', data=df, palette='Set2')
plt.title('Retweets vs. Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Retweets')

plt.tight_layout()
plt.show()


# Group the dataframe by 'Final Sentiments' and calculate summary statistics for 'Likes' and 'Retweets'
summary_stats_likes = df.groupby('Final Sentiments')['Likes'].describe()
summary_stats_retweets = df.groupby('Final Sentiments')['Retweets'].describe()

# Print the summary statistics for Likes
print("Summary statistics for Likes:")
print(summary_stats_likes)
print()

# Print the summary statistics for Retweets
print("Summary statistics for Retweets:")
print(summary_stats_retweets)
