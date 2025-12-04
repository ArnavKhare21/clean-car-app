import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# 1. Load the data
car = pd.read_csv('Cleaned_Car_data.csv')

# 2. Separate features and target
X = car.drop(columns='Price')
y = car['Price']

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Create the OneHotEncoder (to handle text data)
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# 5. Create a transformer
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# 6. Create the Linear Regression Model
lr = LinearRegression()

# 7. Make a pipeline (Transformer -> Model)
pipe = make_pipeline(column_trans, lr)

# 8. Train the model
pipe.fit(X_train, y_train)

# 9. Save the new model to a file
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

print("Success! A new LinearRegressionModel.pkl has been generated.")