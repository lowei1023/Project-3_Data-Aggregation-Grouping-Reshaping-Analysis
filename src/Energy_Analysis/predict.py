"""

Module Name: predict
This module is for the model building part for energy analysis

"""

import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score

class EnergyPredict:
    """
    
    The EnergyPredict1 class encapsulates a Linear Regression-based machine learning model designed 
    for forecasting energy consumption. Its initialization method, __init__, loads data, separates 
    features and the target variable, and triggers the preprocess_data_column_transformer method to 
    set up a preprocessing pipeline using ColumnTransformer. The train_model method fits the entire 
    pipeline (including preprocessing) to the training dataset. Additionally, methods such as 
    plot_predicted_vs_actual_values, residual_plot, distribution_plot, and learning_curve conduct 
    diverse visualizations and analyses on the model's performance. Finally, the linear_regression method 
    executes model training, prediction on test data, computes evaluation metrics (MAE and R-squared), 
    and presents the outcomes.
    
    Attributes:
    - self.data: This attribute holds the dataset loaded from the provided file using pd.read_csv(filename). 
      It's a Pandas DataFrame containing all the data.

    - self.X: Represents the feature matrix. It's a DataFrame containing specific columns ('1_coal', 
      '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind') extracted from self.data. These columns are 
      used as features for the machine learning model.

    - self.y: Signifies the target variable. It's a Pandas Series consisting of the '7_consumption' 
      column extracted from self.data. This column represents the target variable that the model aims 
      to predict.

    - self.X_train, self.X_test, self.y_train, self.y_test: These attributes store the train-test split 
      of the feature matrix (self.X) and target variable (self.y). They are initialized as None in the constructor 
      and are populated once the train_test_split() method is called.

    - self.preprocessor: This attribute holds the preprocessing pipeline. It's initialized as None and 
      later populated by the preprocess_data_column_transformer() method, containing imputation and scaling 
      transformations.

    - self.pipeline: Represents the complete machine learning pipeline, including data preprocessing and 
      the Linear Regression model. Similar to self.preprocessor, it's initialized as None and filled in the 
      preprocess_data_column_transformer() method.

    """

    def __init__(self, filename):
        """Initialize EnergyPredict1 object.

        Parameters:
        filename (str): The path to the CSV file containing energy consumption data.
        """
        # Read the dataset from the provided CSV file and prepare the features (X) and target variable (y)
        self.data = pd.read_csv(filename)
        self.X = self.data[['1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind']]
        self.y = self.data['7_consumption']
        
        # Split the dataset into training and testing sets using train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # Preprocess the data using the specified column transformer
        self.preprocess_data_column_transformer()

    def preprocess_data_column_transformer(self):
        """Preprocess the data using a ColumnTransformer.

        It applies imputation for missing values and scales the numerical features.
        """
        # Define numeric features and create a numeric transformer pipeline
        numeric_features = ['1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        # Create a ColumnTransformer to apply the numeric transformer to specific features
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])

        # Construct a pipeline combining the preprocessor and a LinearRegression model
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())])

    def train_model(self):
        """Train the model on the training data."""
        # Fit the pipeline (including preprocessing) on the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def plot_predicted_vs_actual_values(self):
        """Plot predicted vs. actual values to visualize the model's performance."""
        # Make predictions on the test set and create a scatter plot
        y_pred = self.pipeline.predict(self.X_test)
        plt.scatter(self.y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Comparison between Predicted and Actual Values")
        plt.show()

        # ... (previous code remains the same)

    def residual_plot(self):
        """Generate a residual plot to evaluate the model's residuals.

        It shows the difference between predicted values and actual values.
        """
        # Make predictions on the test set
        y_pred = self.pipeline.predict(self.X_test)

        # Calculate residuals
        residuals = self.y_test - y_pred

        # Create a scatter plot of predicted values against residuals
        plt.scatter(y_pred, residuals)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Residual Plot")
        plt.show()

    def distribution_plot(self):
        """Plot the distribution of actual vs. predicted values.

        It displays the density distribution of actual and predicted values.
        """
        # Make predictions on the test set
        y_pred = self.pipeline.predict(self.X_test)

        # Plot the distribution using seaborn's kdeplot
        sns.kdeplot(self.y_test, label="Actual Values", fill=True)
        sns.kdeplot(y_pred, label="Predicted Values", fill=True)
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.title("Distribution Plot")
        plt.legend()
        plt.show()

    def learning_curve(self):
        """Generate a learning curve to visualize the model's performance.

        It plots the training and validation errors across different training set sizes.
        """
        # Compute the learning curve scores
        train_sizes, train_scores, test_scores = learning_curve(
            self.pipeline, self.X, self.y, cv=5, scoring='neg_mean_squared_error')

        # Calculate mean scores
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        # Plot the learning curve
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, test_scores_mean, label='Validation error')
        plt.xlabel('Training set size')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    def linear_regression(self):
        """Perform linear regression and evaluate the model's performance.

        Returns:
        float: Mean Absolute Error (MAE) of the model.
        float: R-squared Score of the model.
        """
        # Train the model
        self.train_model()

        # Make predictions on the test set
        y_pred = self.pipeline.predict(self.X_test)

        # Calculate Mean Absolute Error (MAE) and R-squared Score
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Print the evaluation metrics
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")


 
