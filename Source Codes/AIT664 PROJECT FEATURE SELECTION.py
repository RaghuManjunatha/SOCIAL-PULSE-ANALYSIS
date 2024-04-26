import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Preprocessed_dataset.csv")

# Encode 'Platform' into numerical values
label_encoder = LabelEncoder()
df['Platform_Encoded'] = label_encoder.fit_transform(df['Platform'])

# Encode 'Final Sentiments' into numerical values if it's categorical
if df['Final Sentiments'].dtype == 'object':
    df['Final_Sentiments_Encoded'] = label_encoder.fit_transform(df['Final Sentiments'])
else:
    df['Final_Sentiments_Encoded'] = df['Final Sentiments']

# Encode 'Season' into numerical values
df['Season_Encoded'] = label_encoder.fit_transform(df['Season'])


"""
Calculate chi-square statistic
"""

# Calculate chi-square statistic and p-value for each categorical feature
chi2_stats = {}
for feature in ['Platform', 'Country', 'Season_Encoded']:
    contingency_table = pd.crosstab(df[feature], df['Final Sentiments'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    chi2_stats[feature] = {'chi2': chi2, 'p_value': p_value}

# Select features with significant p-values
significant_features = [feature for feature, stats in chi2_stats.items() if stats['p_value'] < 0.05]
print("Significant features:", significant_features)


"""
correlation between engagement metrics, platform, and Final Sentiments
"""

# Calculate correlation between engagement metrics, platform, and Final Sentiments
correlation_matrix = df[['Retweets', 'Likes', 'Platform_Encoded', 'Final_Sentiments_Encoded']].corr()

# Select features with high correlation coefficients
relevant_features = correlation_matrix['Final_Sentiments_Encoded'].abs().sort_values(ascending=False).index.tolist()
print("Relevant features:", relevant_features)


"""
Chi-square Test for 'Country'
"""

# Chi-square Test for 'Country'
contingency_table_country = pd.crosstab(df['Country'], df['Final Sentiments'])
chi2_country, p_value_country, _, _ = chi2_contingency(contingency_table_country)

# Chi-square Test for 'Season_Encoded'
contingency_table_season = pd.crosstab(df['Season_Encoded'], df['Final Sentiments'])
chi2_season, p_value_season, _, _ = chi2_contingency(contingency_table_season)

# Print results
print("\nChi-square Test Results:")
print("Significant feature: 'Country'")
print("Chi-square statistic for Country:", chi2_country)
print("P-value for Country:", p_value_country)

print("\nChi-square Test for Season:")
print("P-value for Season_Encoded:", p_value_season)

print("\nCorrelation Analysis Results:")
print("Correlation coefficients:")
print(correlation_matrix)



import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = df[['Retweets', 'Likes', 'Platform_Encoded', 'Final_Sentiments_Encoded']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()






import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, f_oneway
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/FINAL PRESENTATION/Preprocessed_dataset.csv")

# Define functions for statistical tests

# Chi-square Test for Categorical Variables
def chi_square_test(df, feature, target):
    contingency_table = pd.crosstab(df[feature], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value

# Pearson Correlation Coefficient
def pearson_correlation(df, feature, target):
    # Replace infinite values with NaN, then drop rows with NaN values
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, target], how='any')
    # Convert feature to numeric if it's not already
    if df_cleaned[feature].dtype == 'object':
        df_cleaned[feature] = pd.to_numeric(df_cleaned[feature], errors='coerce')
    # Convert target to numeric if it's not already
    if df_cleaned[target].dtype == 'object':
        df_cleaned[target] = pd.to_numeric(df_cleaned[target], errors='coerce')
    corr, p_value = pearsonr(df_cleaned[feature], df_cleaned[target])
    return corr, p_value

# ANOVA Test
def anova_test(df, feature, target):
    groups = [group for _, group in df.groupby(feature)[target]]
    f_statistic, p_value = f_oneway(*groups)
    return f_statistic, p_value

# Perform Tests
chi2, p_chi2 = chi_square_test(df, 'Platform', 'Final Sentiments')
corr, p_pearson = pearson_correlation(df, 'Likes', 'Final Sentiments')
f_stat, p_anova = anova_test(df, 'Country', 'Likes')

# Print results
print("Chi-square Test for 'Platform' and 'Final Sentiments':")
print("Chi-square value:", chi2)
print("P-value:", p_chi2)
print()

print("Pearson Correlation Coefficient between 'Likes' and 'Final Sentiments':")
print("Correlation coefficient:", corr)
print("P-value:", p_pearson)
print()

print("ANOVA Test for 'Country' and 'Likes':")
print("F-statistic:", f_stat)
print("P-value:", p_anova)





# Define the columns to test for significance
columns_to_test = ['Platform', 'Year', 'Month', 'Day', 'Hour', 'Weekday']

# Perform chi-square test for categorical variables and ANOVA for continuous variables
for column in columns_to_test:
    if df[column].dtype == 'object':
        # Chi-square test for categorical variables
        contingency_table = pd.crosstab(df[column], df['Final Sentiments'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-square Test for '{column}':")
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p_value}")
    else:
        # ANOVA test for continuous variables
        group_values = [df[df['Final Sentiments'] == sentiment][column] for sentiment in df['Final Sentiments'].unique()]
        f_statistic, p_value = f_oneway(*group_values)
        print(f"ANOVA Test for '{column}':")
        print(f"F-statistic: {f_statistic}")
        print(f"P-value: {p_value}")





