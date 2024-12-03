#Mikateko Petronella Baloyi
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import datetime
#from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

#Load the Dataset
df = pd.read_csv('C:/Users/mihlotib/Desktop/Mika/Internship_Oasis Infobyte/car data.csv')
df.head()

print(df.columns)

#Exploration of the dataset
print(df.head())
print(df.info())
# Describe categorical columns
print(df.describe(include='O'))

# check null value
null_values = df.isna().sum()
print(null_values)
#counting car Models
print(df['Car_Name'].value_counts())

# identifying top 4 popular car models
top4 = df['Car_Name'].value_counts().reset_index()
cnt = top4[top4['count'] > 4].count()
cnt

# Setting a style for all plots
#sns.set(style='dark')
sns.set(style='whitegrid')

# Distribution of Selling Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Selling_Price'], kde=True, bins=30)
plt.title('Distribution of Selling Price',fontsize=16, color='darkblue')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

# Car Age vs Selling Price
# Calculate car age from the 'Year' column
df['Car_Age'] = 2024 - df['Year']  # Assuming the current year is 2024
print(df.columns)

#Checking if the calculation worked
print(df.head())
#______________________________________________________
# calculating  the number of cars...
for col in df.columns:

    # Only process columns with an object (categorical) data type
    if df[col].dtype != 'object':
        continue

    # Print the value counts of the categorical column
    print(f"Value counts for '{col}':\n{df[col].value_counts()}\n")

    print(df['Owner'].value_counts())
    #_____________________________________

    # features and labels
    X = df.drop('Selling_Price', axis=1)
    Y = df['Selling_Price']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#trying another way
    # c_cols = X_train.select_dtypes(include='O').columns.to_list()
    # n_cols = X_train.select_dtypes(exclude='O').columns.to_list()
    # print(c_cols)
    # print(n_cols)

    # get the numeric and categorical columns
    numeric_features = X.select_dtypes(exclude=['object']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # out of all the numeric features, get only the continuous ones
    for col in numeric_features:
        print(f"{col} := {len(df[col].value_counts())}\n")

    # Set a threshold for unique values (e.g., columns with fewer than 10 unique values are likely categorical)
    unique_value_threshold = 10

#Identify continuous numeric columns
continuous_numeric_cols = [col for col in numeric_features if df[col].nunique() >= unique_value_threshold]

num_columns = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms']
numerical_df = df[num_columns]
corr_matrix = numerical_df.corr()
#heat map
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu')
plt.title('Correlation Heatmap (Numerical Columns)',fontsize=16, color='navy')
plt.show()
#________________________________________________
num_features = ['Year', 'Driven_kms', 'Selling_Price', 'Present_Price']
num_data = df[num_features]
for feature in num_data:
    plt.figure(figsize=(10, 8))
    sns.displot(data=df, x=feature, kde=True)
    plt.title(f'Distribution of: {feature}',fontsize=15, color='skyblue')
    plt.tight_layout()
    plt.show()
#_____________________
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

for feature in categorical_features:
    plt.figure(figsize=(10, 8))
    sns.countplot(x=feature, data=df)
    plt.title(f'Frequency of {feature}')
    plt.xlabel(f'{feature}', fontsize=14, color='skyblue')
    plt.ylabel('Frequency', fontsize=14, color='skyblue')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plt.figure(figsize=(10, 8))
sns.scatterplot(x='Car_Age', y='Selling_Price', data=df)
plt.title('Car Age vs Selling Price',fontsize=16, color='navy')
plt.xlabel('Car Age (years)')
plt.ylabel('Selling Price')
plt.show()

# Present Price vs Selling Price
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=df)
plt.title('Present Price vs Selling Price',fontsize=16, color='navy')
plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.show()

# Driven_kms vs Selling Price
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=df)
plt.title('Kilometers Driven vs Selling Price',fontsize=16, color='navy')
plt.xlabel('Driven (kms)')
plt.ylabel('Selling Price')
plt.show()

# Separate the features and target variable
X = df.drop(columns=['Selling_Price'])  # Features
y = df['Selling_Price']

# Identify categorical and numerical columns
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define models to test
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', LinearRegression())]),

    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Store results
results[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}
print(f"Results for {model_name}:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}\n")

# # Initialize the Random Forest Regressor
 #rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # Train the model
# rf_regressor.fit(X_train, y_train)
#__________________________________

X = df[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
y = df['Selling_Price']

# Handle categorical features using one-hot encoding (if needed)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure feature names match between training and testing datasets
missing_features = set(X_train.columns) - set(X_test.columns)
for feature in missing_features:
    X_test[feature] = 0

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Testing the model with custom data
custom_data = pd.DataFrame({
    'Year': [2016],
    'Present_Price': [10.0],
    'Driven_kms': [70000],
    'Owner': [0],
    'Fuel_Type_CNG': [0],
    'Fuel_Type_Diesel': [0],
    'Fuel_Type_Petrol': [1],
    'Selling_type_Dealer': [0],
    'Selling_type_Individual': [1],
    'Transmission_Automatic': [0],
    'Transmission_Manual': [1]
})
custom_prediction = rf_regressor.predict(custom_data)

print(f"Predicted Selling Price for Custom Data: {custom_prediction[0]:.2f}")
#______________________________


import joblib

# Save the model
joblib.dump(model, 'car_price_model.pkl')

top_car = df.groupby('Car_Name')['Present_Price'].mean().nlargest(10)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_car.values, y=top_car.index, palette='viridis')
plt.title(f'Top {10} Car Models by Mean Present_Price',fontsize=16, color='navy')
plt.xlabel('Mean Present_Price')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()

sns.lmplot(x='Year', y='Selling_Price', data=df)
plt.show()
#__________________________________________


