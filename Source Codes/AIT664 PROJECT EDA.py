import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Preprocessed_dataset.csv")

#EDA STARTS HERE
# Count the frequency of each sentiment
sentiment_counts = df['Final Sentiments'].value_counts()

# Plot the distribution of sentiment
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(rotation= 0, ha='right')
plt.show()


# Plot bar plot for platform distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Platform', data=df, color='skyblue')
plt.title('Platform Distribution')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Filter the dataframe for only Positive, Negative, and Neutral sentiments
filtered_df = df[df['Final Sentiments'].isin(['Positive', 'Negative', 'Neutral'])]

# Plot sentiment over time (e.g., monthly)
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Sentiment', hue='Final Sentiments', data=filtered_df)
plt.title('Sentiment Variation Over Time')
plt.xlabel('Month')
plt.ylabel('')  # Remove y-axis label
plt.yticks([])  # Remove y-axis ticks and labels
plt.xticks(range(1, 13))  # Display all months on the x-axis
plt.xticks(rotation=0)
plt.legend(title='Final Sentiments')
plt.show()


# Box plot of sentiment vs. engagement rate
plt.figure(figsize=(10, 6))
sns.boxplot(x='Final Sentiments', y='Engagement Rate', hue='Platform', data=df)
plt.title('Sentiment vs. Engagement Rate')
plt.xlabel('Sentiment')
plt.ylabel('Engagement Rate')
plt.xticks(rotation=0)
plt.legend(title='Platform')
plt.show()


# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Encode 'Final Sentiments' column
df['Sentiment_Encoded'] = label_encoder.fit_transform(df['Final Sentiments'])
# Calculate the correlation matrix
correlation_matrix = df[['Sentiment_Encoded', 'Retweets', 'Likes']].corr()
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sentiment_Encoded', 'Retweets', 'Likes']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Sentiment and Engagement Metrics')
plt.show()
# Print the correlation values
print("Correlation between Sentiment and Engagement Metrics:")
print(correlation_matrix)


################################################################################################

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'Final Sentiments' column
df['Sentiment_Encoded'] = label_encoder.fit_transform(df['Final Sentiments'])

# Display summary statistics
summary_stats = df.describe(include='all')
print("Summary Statistics:")
print(summary_stats)

# Display data types and non-null values
info = df.info()
print("\nData Types and Non-null Values:")
print(info)

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

##########################################################################

#EDA WITH WEEKDAYS

plt.figure(figsize=(8, 6))
sns.countplot(x='Weekday', data=df)
plt.title('Count of Tweets by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Weekday', hue='Final Sentiments', data=df)
plt.title('Sentiment Distribution by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Weekday', hue='Platform', data=df)
plt.title('Platform Distribution by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.legend(title='Platform')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Weekday', y='Engagement Rate', data=df)
plt.title('Engagement Rate by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Engagement Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Weekday', y='Retweets', data=df, label='Retweets')
sns.lineplot(x='Weekday', y='Likes', data=df, label='Likes')
plt.title('Retweets and Likes Over Weekdays')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.legend()
plt.show()

sentiment_weekday = df.groupby(['Final Sentiments', 'Weekday']).size().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(sentiment_weekday, annot=True, cmap='YlGnBu')
plt.title('Sentiment Distribution by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Sentiment')
plt.show()


########################################################################
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

