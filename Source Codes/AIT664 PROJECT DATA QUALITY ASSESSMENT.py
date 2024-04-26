import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Preprocessed_dataset.csv")

# 1. Completeness
# Check for missing values
missing_values = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
completeness = 1 - (missing_values / total_cells)

# 2. Accuracy
# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
accuracy = 1 - (duplicate_rows / df.shape[0])

# 3. Relevance (Assuming relevance refers to the importance of features)
# Assess the relevance of columns based on domain knowledge
relevant_columns = ['Text', 'Final Sentiments', 'Timestamp', 'User', 'Platform', 'Retweets', 'Likes', 'Country', 'Weekday']

# 4. Validity
# Check for anomalies or inconsistencies in specific columns
# For example, check the validity of the 'Final Sentiments' column
valid_sentiments = ['Positive', 'Negative', 'Neutral']
validity_column_name = 'Final Sentiments'  # Update with the correct column name if needed
if validity_column_name in df.columns:
    validity = df[validity_column_name].isin(valid_sentiments).all()
else:
    validity = False

# Print assessment results
print("Data Quality Assessment:")
print(f"Completeness: {completeness:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print("Relevance: Columns considered relevant - ", relevant_columns)
if validity:
    print(f"Validity of '{validity_column_name}' column: {validity}")
else:
    print(f"'{validity_column_name}' column not found in the dataset.")
