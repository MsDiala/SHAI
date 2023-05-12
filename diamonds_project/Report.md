# Diamond Price Prediction Analysis

We conducted an analysis to predict diamond prices using two models: Linear Regression and Random Forest. The dataset consisted of various features such as carat, clarity, color, depth, table, cut, and dimensions of the diamond (x, y, z). Our objective was to evaluate the performance of the models and determine the most important feature affecting the model predictions.

## Model Performance

### - Linear Regression

- RMSE: 1349.41
- Accuracy: 0.8889

### - Random Forest

- RMSE: 561.81
- Accuracy: 0.9808

Based on the evaluation metrics, the Random Forest model outperformed the Linear Regression model in terms of both RMSE and accuracy. It achieved a significantly lower RMSE value of 561.81 and a higher accuracy of 0.9808 compared to the Linear Regression model's RMSE of 1349.41 and accuracy of 0.8889.

## Feature Importance

We also analyzed the importance of the features in the Random Forest model. The feature importance scores indicate the relative influence of each feature on the model predictions. The results are as follows:

- Carat: 0.5402
- Y: 0.3485
- Clarity: 0.0651
- Color: 0.0278
- Z: 0.0057
- X: 0.0055
- Depth: 0.0034
- Table: 0.0024
- Cut: 0.0013

According to the Random Forest model, the most important feature affecting the diamond price prediction is "carat" with a feature importance score of 0.5402. This indicates that carat has the highest influence on the model's predictions. The "y" feature also carries significant importance with a score of 0.3485. The remaining features, including clarity, color, z, x, depth, table, and cut, have relatively lower importance scores.