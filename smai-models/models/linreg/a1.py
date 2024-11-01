import numpy as np
import pandas as pd
from linreg import PolyRegression
import os
import matplotlib.pyplot as plt

df = pd.read_csv(open("../../data/external/linreg.csv", "r"))

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

## Part 1

# plt.plot(x_train, y_train, 'o')
   # # Append 1 to x_vals
    # x_train = np.vander(x_train, degree+1)
    # x_val = np.vander(x_val, degree+1)
    # x_test = np.vander(x_test, degree+1)

    # Multiply x_vals with the coefficients to obtain predicted values
    # predicted_train = np.dot(x_train, coeffs)
    # predicted_val = np.dot(x_val, coeffs)
    # predicted_test = np.dot(x_test, coeffs)

    # # Plot the predicted values against x_vals
    # plt.plot(x_train[:, 0], predicted_train, label='Line of Best Fit (Train)')
    # plt.plot(x_val[:, 0], predicted_val, label='Line of Best Fit (Validation)')
    # plt.plot(x_test[:, 0], predicted_test, label='Line of Best Fit (Test)')

    # # Plot the original values
    # plt.plot(x_train[:, 0], y_train, 'o', label='Original Values (Train)')
    # plt.plot(x_val[:, 0], y_val, 'o', label='Original Values (Validation)')
    # plt.plot(x_test[:, 0], y_test, 'o', label='Original Values (Test)')

    # # Add legend
    # plt.legend()

    # # Show the plot
    # plt.show()

## Part 2
# MSE_MIN = float('inf')
# MSE_MIN_DEGREE = 0

# for degree in range(1, 50):
#     learning_rate = 0.0001/pow(3,degree)

#     model = PolyRegression(x_train, y_train, degree)
#     # Fit the linear model
#     coeffs = model.fit(learning_rate=learning_rate)

#     mse_train = model.MSE(x_train, y_train, coeffs)
#     mse_val = model.MSE(x_val, y_val, coeffs)
#     mse_test = model.MSE(x_test, y_test, coeffs)

#     sd_train = model.sd(x_train, y_train, coeffs)
#     sd_val = model.sd(x_val, y_val, coeffs)
#     sd_test = model.sd(x_test, y_test, coeffs)

#     var_train = model.variance(x_train, y_train, coeffs)
#     var_val = model.variance(x_val, y_val, coeffs)
#     var_test = model.variance(x_test, y_test, coeffs)

 

#     # Printing relevant metrics
#     print("Degree:", degree)
#     print("Learning Rate:", learning_rate)
#     print("Coefficients:", coeffs)
#     print("MSE (Train):", mse_train)
#     print("MSE (Validation):", mse_val)
#     print("MSE (Test):", mse_test)
#     print("SD (Train):", sd_train)
#     print("SD (Validation):", sd_val)
#     print("SD (Test):", sd_test)
#     print("Variance (Train):", var_train)
#     print("Variance (Validation):", var_val)
#     print("Variance (Test):", var_test)
#     print("\n")

#     if mse_test < MSE_MIN:
#         MSE_MIN = mse_test
#         MSE_MIN_DEGREE = degree
#         best_coeffs = coeffs

# print("Minimum MSE:", MSE_MIN)
# print("Degree with Minimum MSE:", MSE_MIN_DEGREE)
# print("Coefficients with Minimum MSE:", best_coeffs)


## Part 3
degree = 5
model = PolyRegression(x_train, y_train, degree)
coeff_list = model.fit(learning_rate=0.03)
y_vals = y_test
x_vals = x_test
# print(coeff_list)

def create_and_save_plots(num_plots, save_dir, degree, x_vals, y_vals, coeff_list):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(num_plots):
        coeffs = coeff_list[i]
        # Generate Vandermonde matrix
        X = np.vander(x_test, degree+1)
        
        # Predict y values
        y_pred = np.dot(X, coeffs)
        
        MSE = model.MSE(x_vals, y_vals, coeffs)
        SD = model.sd(x_vals, y_vals, coeffs)
        Variance = model.variance(x_vals, y_vals, coeffs)

        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        # Plot the actual and predicted values
        plt.scatter(x_vals, y_vals, color='blue', label='Actual')
        plt.plot(x_vals, y_pred, 'r--', label='Predicted')
        
        # Add lines joining actual and predicted values every 10 datapoints
        # for j in range(0, len(x_vals), 10):
        #     plt.plot([x_vals[j], x_vals[j]], [y_vals[j], y_pred[j]], 'g-', linewidth=0.5)
        
        # Add MSE, SD, and Variance to the legend
        legend_text = f'MSE: {MSE:.2f}, SD: {SD:.2f}, Variance: {Variance:.2f}'
        plt.legend([legend_text])
        
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'Plot {i+1}')
        
        # Save the plot
        filename = os.path.join(save_dir, f'plot_{i+1}.png')
        plt.savefig(filename)
        
        # Close the figure to free up memory
        plt.close()
        
        print(f"Saved plot {i+1} to {filename}")

# Specify the number of plots and the directory to save them
num_plots = 100
save_directory = "plots/5"


# Call the function to create and save the plots
create_and_save_plots(num_plots, save_directory, degree, x_vals, y_vals, coeff_list)

