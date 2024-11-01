import numpy as np
import pandas as pd
from linreg import PolyRegression
import os
import matplotlib.pyplot as plt

df = pd.read_csv(open("../../data/external/regularisation.csv", "r"))

x_vals = np.array([df["x"].values]).T
x_vals = x_vals.reshape(-1)
y_vals = np.array(df["y"].values)
degree = 2
learning_rate = 0.0001/(degree**2)

# Randomly split the data into train, validation, and test sets
train_indices = df.sample(frac=0.8, random_state=42).index
val_indices = df.drop(train_indices).sample(frac=0.5, random_state=42).index
test_indices = df.drop(train_indices).drop(val_indices).index

x_train = df.loc[train_indices, "x"]
y_train = df.loc[train_indices, "y"]
x_val = df.loc[val_indices, "x"]
y_val = df.loc[val_indices, "y"]
x_test = df.loc[test_indices, "x"]
y_test = df.loc[test_indices, "y"]


## Part 4.1
for l1 in [1, 0.1, 0.01]:
    MSE_MIN = float('inf')
    MSE_MIN_DEGREE = 0
    for degree in range(1, 20):
        learning_rate = 0.0001/pow(3,degree)

        model = PolyRegression(x_train, y_train, degree)
        # Fit the linear model
        coeffs = model.fit(learning_rate=learning_rate, l1=l1)

        mse_train = model.MSE(x_train, y_train, coeffs)
        mse_val = model.MSE(x_val, y_val, coeffs)
        mse_test = model.MSE(x_test, y_test, coeffs)

        sd_train = model.sd(x_train, y_train, coeffs)
        sd_val = model.sd(x_val, y_val, coeffs)
        sd_test = model.sd(x_test, y_test, coeffs)

        var_train = model.variance(x_train, y_train, coeffs)
        var_val = model.variance(x_val, y_val, coeffs)
        var_test = model.variance(x_test, y_test, coeffs)

        if mse_test < MSE_MIN:
            MSE_MIN = mse_test
            MSE_MIN_DEGREE = degree
            best_coeffs = coeffs

    print("Lambda:", l1)
    print("Minimum MSE:", MSE_MIN)
    print("Degree with Minimum MSE:", MSE_MIN_DEGREE)
    print("Coefficients with Minimum MSE:", best_coeffs)
    print("\n")

## Part 4.2
# for l2 in [1, 0.1, 0.01]:
#     MSE_MIN = float('inf')
#     MSE_MIN_DEGREE = 0
#     for degree in range(1, 20):
#         learning_rate = 0.0001/pow(3,degree)

#         model = PolyRegression(x_train, y_train, degree)
#         # Fit the linear model
#         coeffs = model.fit(learning_rate=learning_rate, l2=l2)

#         mse_train = model.MSE(x_train, y_train, coeffs)
#         mse_val = model.MSE(x_val, y_val, coeffs)
#         mse_test = model.MSE(x_test, y_test, coeffs)

#         sd_train = model.sd(x_train, y_train, coeffs)
#         sd_val = model.sd(x_val, y_val, coeffs)
#         sd_test = model.sd(x_test, y_test, coeffs)

#         var_train = model.variance(x_train, y_train, coeffs)
#         var_val = model.variance(x_val, y_val, coeffs)
#         var_test = model.variance(x_test, y_test, coeffs)

#         if mse_test < MSE_MIN:
#             MSE_MIN = mse_test
#             MSE_MIN_DEGREE = degree
#             best_coeffs = coeffs

#     print("Lambda:", l2)
#     print("Minimum MSE:", MSE_MIN)
#     print("Degree with Minimum MSE:", MSE_MIN_DEGREE)
#     print("Coefficients with Minimum MSE:", best_coeffs)
#     print("\n")

