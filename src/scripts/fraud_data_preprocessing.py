import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fraud_data = pd.read_csv('Fraud_Data.csv')
creditcard_data = pd.read_csv('creditcard.csv')
ip_data = pd.read_csv('IpAddress_to_Country.csv')

# 1. Handle Missing Values
def handle_missing_values(df, dataset_name):
    print(f"\nMissing values in {dataset_name}:")
    print(df.isnull().sum())
    
    # For fraud_data, drop rows with missing critical fields
    if dataset_name == 'Fraud_Data':
        df = df.dropna(subset=['user_id', 'purchase_time', 'purchase_value'])
        # Impute missing categorical with mode
        for col in ['source', 'browser', 'sex']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        # Impute missing age with median
        df['age'] = df['age'].fillna(df['age'].median())
    elif dataset_name == 'creditcard':
        # For creditcard, drop rows with missing Amount or Class
        df = df.dropna(subset=['Amount', 'Class'])
    
    return df

fraud_data = handle_missing_values(fraud_data, 'Fraud_Data')
creditcard_data = handle_missing_values(creditcard_data, 'creditcard')

# 2. Data Cleaning
# Remove duplicates
fraud_data = fraud_data.drop_duplicates()
creditcard_data = creditcard_data.drop_duplicates()

# Correct data types
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['user_id'] = fraud_data['user_id'].astype(int)
fraud_data['purchase_value'] = fraud_data['purchase_value'].astype(float)
fraud_data['age'] = fraud_data['age'].astype(int)

# 3. Exploratory Data Analysis (EDA)
def perform_eda(df, dataset_name):
    # Univariate Analysis
    print(f"\nEDA for {dataset_name}")
    
    # Numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{col}_distribution.png')
        plt.close()
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.savefig(f'{col}_count.png')
        plt.close()
    
    # Bivariate Analysis (with respect to Class)
    if 'class' in df.columns:
        for col in numerical_cols:
            if col != 'class':
                plt.figure(figsize=(8, 4))
                sns.boxplot(x='class', y=col, data=df)
                plt.title(f'{col} vs Class')
                plt.savefig(f'{col}_vs_class.png')
                plt.close()

perform_eda(fraud_data, 'Fraud_Data')
perform_eda(creditcard_data, 'creditcard')

# 4. Merge Datasets for Geolocation Analysis
def ip_to_int(ip):
    try:
        parts = ip.split('.')
        return sum(int(part) * (256 ** (3 - i)) for i, part in enumerate(parts))
    except:
        return np.nan

fraud_data['ip_int'] = fraud_data['ip_address'].apply(ip_to_int)

def map_ip_to_country(ip_int, ip_data):
    try:
        if np.isnan(ip_int):
            return 'Unknown'
        match = ip_data[(ip_data['lower_bound_ip_address'] <= ip_int) & 
                       (ip_data['upper_bound_ip_address'] >= ip_int)]
        return match['country'].iloc[0] if not match.empty else 'Unknown'
    except:
        return 'Unknown'

fraud_data['country'] = fraud_data['ip_int'].apply(lambda x: map_ip_to_country(x, ip_data))

# 5. Feature Engineering for Fraud_Data
# Transaction frequency and velocity
user_purchase_counts = fraud_data.groupby('user_id').size().reset_index(name='transaction_frequency')
fraud_data = fraud_data.merge(user_purchase_counts, on='user_id')

# Time-based features
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600

# 6. Data Transformation
# Handle Class Imbalance
def handle_class_imbalance(X, y):
    print("\nClass distribution before SMOTE:")
    print(pd.Series(y).value_counts())
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    
    return X_resampled, y_resampled

# Prepare features for fraud_data
X_fraud = fraud_data.drop(['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_int'], axis=1)
y_fraud = fraud_data['class']

# Encode categorical features
categorical_cols = ['source', 'browser', 'sex', 'country']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X_fraud[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_fraud = X_fraud.drop(categorical_cols, axis=1).reset_index(drop=True)
X_fraud = pd.concat([X_fraud, X_encoded], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Apply SMOTE only to training data
X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

# Normalization and Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Save processed data
processed_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
processed_data['class'] = y_train_resampled
processed_data.to_csv('processed_fraud_data_train.csv', index=False)

test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_data['class'] = y_test.reset_index(drop=True)
test_data.to_csv('processed_fraud_data_test.csv', index=False)

print("\nPreprocessing complete. Processed data saved as 'processed_fraud_data_train.csv' and 'processed_fraud_data_test.csv'")