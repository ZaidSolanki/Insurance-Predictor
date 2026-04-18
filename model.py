import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# dataset load
data = pd.read_csv("insurance_data_linear.csv")

# features & target
X = data.drop("charges", axis=1)
y = data["charges"]

# categorical convert
X = pd.get_dummies(X, drop_first=True)

# model train
model = LinearRegression()
model.fit(X, y)

# save model
pickle.dump((model, X.columns), open("model.pkl", "wb"))

print("Model trained successfully")