import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("C:/Users/raghu/Desktop/AIT 664/AIT 664 FINAL PROJECT/PROJECT UPDATE 2/Preprocessed_dataset.csv")

# Split the dataset into features (X) and target (y)
columns_to_exclude = ['Text']
X = df.drop(columns=['Sentiment', 'Timestamp', 'User', 'Platform', 'Primary Hash Tag', 'Secondary hashtag', 'Final Sentiments', 'Season'] + columns_to_exclude)
y = df['Final Sentiments']

# Encode categorical variables (if any) using one-hot encoding
categorical_features = ['Country']  # Adjust this list if 'Country' column exists
preprocessor = OneHotEncoder(handle_unknown='ignore')
X_encoded = preprocessor.fit_transform(X[categorical_features])

# Extract column names after one-hot encoding
encoded_feature_names = preprocessor.get_feature_names_out(input_features=categorical_features)

# Convert the one-hot encoded features to a DataFrame
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_feature_names)

# Concatenate the one-hot encoded features with the remaining numerical features
X_final = pd.concat([X_encoded_df, X.drop(columns=categorical_features)], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Initialize Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)

# Train the Logistic Regression model
lr_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = lr_classifier.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Initialize Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the Decision Tree model
dt_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the Decision Tree model
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
