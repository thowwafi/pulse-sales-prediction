# Mobile-Credit Seller Prediction

## Backgrounds & Objectives

For many telecom companies, predicting the number of mobile credit seller in an area remains a problem. Since if they could predict the number, they could provide customer service & sales support efficiently on those area.

The goal of the project is to build a model which predict the number of mobile credit seller in an area with at least R2 score of 0.7 and RMSE ~300.

The expected input payload will be data batch from the user of the expected features and its output would be the number of the predicted mobile-credit seller

## Project Architecture

![Project Architecture](/images/project_architecture.png)

## Feature Data Type

![Feature Data Type](/images/feature_data_type.png)

## Check Missing Value

![Check Missing Value](/images/check_missing.png)

## Outlier Checking

![Outlier Checking 1](/images/check_outlier1.png)
![Outlier Checking 2](/images/check_outlier2.png)
![Outlier Checking 3](/images/check_outlier3.png)
![Outlier Checking 4](/images/check_outlier4.png)

## Label Encoding

![Label Encoding](/images/label_encoding.png)

## Feature Correlation Map

![Feature Correlation Map](/images/correlation_map.png)

## Baseline Regression Model

![Linear Regression](/images/linear_regression.png)
![Base Linear Regression](/images/base_linear.png)

## Lasso Regression Model
![Lasso Regression](/images/lasso.png)


## Decision Tree Regression Model
![Decision Tree Regression](/images/decision_tree.png)

## Conclusion

![Conclusion](/images/conclution.png)

## References

- https://towardsdatascience.com/hyperparameter-tuning-in-python-21a76794a1f7
- https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
- http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-6-feature-selection-with-fixed-trainvalidation-splits
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html


