import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Sample dataset (you can replace this with a CSV)
data = {
    'Experience': [1, 3, 5, 7, 9, 11],
    'Education': ['Bachelors', 'Bachelors', 'Masters', 'Masters', 'PhD', 'PhD'],
    'Job Title': ['Engineer', 'Engineer', 'Manager', 'Manager', 'Director', 'Director'],
    'Location': ['Mumbai', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi'],
    'Salary': [40000, 50000, 70000, 80000, 100000, 110000]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Preprocess categorical features
categorical_features = ['Education', 'Job Title', 'Location']
numeric_features = ['Experience']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Display results
print("Predicted salaries:", y_pred)
print("Actual salaries:", y_test.values)

# Visualize
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salaries')
plt.grid(True)
plt.show()
