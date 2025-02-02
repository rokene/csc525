import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# For this assignment we want to use the linear regression model only
# otherwise set to none to test other models
override_model_name = "Baseline Linear"

###############################################################################
# Load the California Housing dataset
# The Boston dataset has been removed in later versions.
###############################################################################
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

###############################################################################
# Define multiple models to compare via cross-validation
###############################################################################
# Baseline Linear Regression
model_baseline = LinearRegression()

# Ridge Pipeline with Standard Scaling and Polynomial Features
model_ridge_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", Ridge(alpha=1.0))
])

# Random Forest Regressor
model_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Dictionary of models to compare
models = {
    "Baseline Linear": model_baseline,
    "Ridge + Polynomial": model_ridge_poly,
    "Random Forest": model_rf
}

###############################################################################
# Cross-validate each model on the training set
###############################################################################
def cross_val_mean_mse(model, X_train, y_train, cv=5):
    """Returns the average MSE (positive value) across CV folds."""
    scores = cross_val_score(model, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=cv)
    mean_mse = -np.mean(scores)  # convert from negative MSE
    return mean_mse

cv_results = {}
for name, model in models.items():
    mse_val = cross_val_mean_mse(model, X_train, y_train, cv=5)
    cv_results[name] = mse_val

# Print cross-validation results
print("==== Cross-Validation Results (Training Split) ====")
for model_name, mse_val in cv_results.items():
    print(f"{model_name} MSE: {mse_val:.4f}")

# Choose the best model (lowest MSE)
best_model_name = min(cv_results, key=cv_results.get)

# Override best model based on MSE to model named
if override_model_name:
  print(f"WARNING: Overrided Model is {override_model_name}")
  best_model_name = override_model_name

best_model = models[best_model_name]

print(f"\nChosen Model for Final Evaluation: {best_model_name}")

###############################################################################
# Fit the chosen model on the entire training set and evaluate on the test
###############################################################################
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("\n==== Final Model Test Set Evaluation ====")
print(f"Mean Squared Error (Test): {mse_test:.4f}")
print(f"R-squared (Test):         {r2_test:.4f}")

###############################################################################
# Compare first 5 predictions with actual values, including % difference
###############################################################################
comparison_list = []
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    pct_diff = ((predicted - actual) / actual) * 100 if actual != 0 else np.nan
    comparison_list.append({
        "Predicted": predicted,
        "Actual": actual,
        "% Diff": pct_diff
    })

comparison_df = pd.DataFrame(comparison_list)
print("\n---- Sample Predictions vs Actual Values ----")
print(comparison_df)

###############################################################################
# Print coefficients if the best model is a plain LinearRegression
###############################################################################
# If it’s the pipeline or random forest, it has a more complex structure (poly features,
# multiple trees, etc.), so we won't try to print their internal coefficients.
if isinstance(best_model, LinearRegression):
    print("\n---- Feature Coefficients (Baseline LinearRegression) ----")
    for feature_name, coef_val in zip(X_train.columns, best_model.coef_):
        print(f"{feature_name}: {coef_val:.5f}")
    print("Intercept:", best_model.intercept_)

###############################################################################
# Prompt the user for new inputs to get predictions until "quit"
###############################################################################
print("\n---- Predict with User Input ----")
print(
    "Enter feature values for a single house in the order:\n"
    "1) MedInc, 2) HouseAge, 3) AveRooms, 4) AveBedrms, "
    "5) Population, 6) AveOccup, 7) Latitude, 8) Longitude.\n"
    "Type 'quit' to exit.\n"
)

while True:
    user_input = input("Enter 8 values separated by spaces (or 'quit' to exit): ")
    if user_input.lower() == "quit":
        print("Exiting the prediction loop. Goodbye!")
        break

    # Attempt to parse 8 floats
    try:
        values = list(map(float, user_input.split()))
        if len(values) != 8:
            print("Error: You must enter exactly 8 numerical values.")
            continue
    except ValueError:
        print("Error: Please enter valid numerical values or 'quit' to exit.")
        continue

    new_data = np.array(values).reshape(1, -1)
    user_pred = best_model.predict(new_data)[0]
    print(f"Predicted median house value: {user_pred:.4f}\n")
