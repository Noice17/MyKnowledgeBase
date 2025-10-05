# 1. Classification
## Machine Learning w Scikit-learn
- ML is the process whereby:
	- computers are given the ability to learn to make decision from data w/o explicitly programming
	- Examples
		- Predicting if email is spam or not spam
		- Clustering books
- Unsupervised learning
	- uncovering hidden patterns from unlabeled data
		- grouping (clustering)
- Supervised Learning
	- the predicted values are known
	* Types
		* **Classification**: target variable consists of categories
		* **Regression**: target variable is continuous
	* Naming Conventions
		* Feature = predictor variable = independent variable
		* Target variable = dependent variable = response variable
* Before using Supervised Learning
	* Requirements
		* no missing values
		* data in numeric format
		* data stored in pandas DataFrame or NumPy arrays
	* Perform Exploratory Data Analysis (EDA) first
* Scikit-learn syntax
```python
from sklearn.module import Model
model = Model()
model.fit(x,y)
predictions = model.predict(X_new)
print(predictions)
```

## The Classification Challenge
Steps.
1. Build a model
2. Model learns from the labeled data we pass to it
3. Pass unlabeled data to the model as input
4. Model predicts the labels of the unseen data

K-Nearest Neighbors
* predict the label of a data point by
	* Looking at the `k` closest labeled data points
	* taking a majority vote

Using scikit-learn to fit a classifier
```python
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values

# (3333,2), (3333,)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X,y)
```
Predicting on unlabeled data
```python
X_new = np.array([
				[56.8, 17.5],
				[24.4, 24.1],
				[50.1, 10.9]])
print(X_new.shape)
# (3,2)

predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
#Predictions: [1,0,0]
```

## Measuring Model Performance
* In classification, `accuracy` used
* Split  data
	* Train set
		* Fit/train classifier on training set
	* Test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### Model complexity
* as k increases, the decision boundary is less effective by individual observations
* Larger k = less complex model = can cause underfitting
* Smaller k = more complex model = can lead to overfitting

```python
train_accuracis = {}
test_accuracies = {}
neighbors = np.arrange(1,26)
for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors = neighbor)
	knn.fit(X_train, y_train)
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
```

Plotting results
```python
plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label = "Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

# 2.  Regression
```python
import pandas as pd
diabetes_df = pd.read_csv("diabetes.csv")
print(diabetes_df.head())
```
## Creating feature and target arrays
```python
X = diabetes_df.drop("glucode",axis = 1).values
y = diabetes_df["glucose"].values
print(type(X), type(y))
```

## Making predictions from a single feature
```python
X_bmi = X[:,3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1,1)
print(X_bmi.shape)
```

## Plotting glucose vs. body mass index
```python
import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)
plt.ylabel()
plt.xlabel
plt.show()
```

## Fitting a regression model
```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
```

## Plotting
```python
# Import matplotlib.pyplot

import matplotlib.pyplot as plt

  

# Create scatter plot

plt.scatter(X, y, color="blue")

  

# Create line plot

plt.plot(X, predictions, color="red")

plt.xlabel("Radio Expenditure ($)")

plt.ylabel("Sales ($)")

  
# Display the plot

plt.show()
```
## The Basics of Linear regression
Regression Mechanics
* y = ax + b
	* Simple linear regression uses one feature
	* y = target
	* x = single feature
	* a,b = parameters/coefficients of the model - slope, intercept
* How do we choose a and b?
	* define an error function for any given line
	* choose the line that minimizes the error function
The Loss function
* residual: vertical distance between the line and datapoints
Ordinary Least Squares: minimizes RSS
Linear regression in higher dimensions
* y = a1x1 + a2x2 + b
* In higher dimensions, known as multiple regression
* Must specify coefficients for each feature and the variable b
	* y = a2x1 + a2x2 +... + an xn + b
### Linear regression using all features
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 43)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
```

### R-squared
default metric: R-square
- quantifies the variance in target values explained by the features
	- values range from 0 to 1
![[r-squared plot.png]]

#### R-squared in scikit-learn
```python
reg_all.score(X_test, y_test)
```

### Mean squared error and root mean squared error
![[MSE formulat.png]]
- MSE is measured in target units, squared
![[RMSE formula.png]]
- Measure RMSE in the same units at the target variable

#### RMSE in scikit-learn
```python
from sklearn.metrics impot root_mean_squared_error
root_mean_squared_error(y_test, y_pred)
```

### Ex: Fit and Predict for Regression
```python
# Create X and y arrays

X = sales_df.drop("sales", axis=1).values

y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  

# Instantiate the model

reg = LinearRegression()

  

# Fit the model to the data

reg.fit(X_train, y_train)

# Make predictions

y_pred = reg.predict(X_test)

print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
```

### Ex. Regression Performance
```python
# Import root_mean_squared_error

from sklearn.metrics import root_mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE

rmse = root_mean_squared_error(y_test, y_pred)

  

# Print the metrics

print("R^2: {}".format(r_squared))

print("RMSE: {}".format(rmse))
```

## Cross-validation
- model performance is dependent on the way we split up the data
- not representative of the model's ability to generalize to unseen data
- Solution: cross-validation
- 5-fold, 10-fold, k-fold
- more folds, more computationally expensive
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits = 6, shuffle = True, random_state = 42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)

# evaluation
print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_result, [0.025, 0.975])) #95# confident interval
```

### Ex: Cross-validation for R-squared
```python
# Import the necessary modules

from sklearn.model_selection import cross_val_score, KFold

  

# Create a KFold object

kf = KFold(n_splits=6, shuffle=True, random_state=5)

  

reg = LinearRegression()

  

# Compute 6-fold cross-validation scores

cv_scores = cross_val_score(reg, X, y, cv=kf)

  

# Print scores

print(cv_scores)
```

### Ex: Analyzing cross-validation metrics
```python
# Print the mean

print(np.mean(cv_results))

  

# Print the standard deviation

print(np.std(cv_results))

  

# Print the 95% confidence interval

print(np.quantile(cv_results, [0.025, 0.975]))
```
## Regularized Regression
- technique used to avoid overfitting
### Ridge regression
- ![[ridge regression formula.png]]
- penalizes large positive or negative coefficients
- alpha: parameter we need to choose
- picking alpha is similar to picking k in KNN
- Hyperparameter: variable used to optimize model parameters
- alpha controls model complexity
	- alpha = 0 = OLS(Can lead to overfitting)
	- very high alpha: Can lead to underfitting
```python
fromt sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
	ridge = Ridge(alpha = alpha)
	ridge.fit(X_train, y_train)
	y_pred = ridge.predict(X_test)
	scores.append(ridge.score(X_test, y_test))
print(scores)
```

### Lasso regression
- ![[Lasso regression formula.png]]
```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
	lasso = Lasso(alpha=alpha)
	lasso.fit(X_train, y_train)
	lasso_pred = lasso.predict(X_test)
	scores.append(lasso.score(X_test, y_test))
print(scores)
```
* can select important features of a dataset
* Shrinks the coefficients of less important features to zero
* features not shrunk to zero are selected by lasso
#### Lasso for Feature Selection in scikit-learn
```python
from sklearn.linear_model import Lasso
X = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values
names = diabetes_df.drop("glucose",axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X,y).coef_
plt.bar(names, lasso_coef)
plt.xticks(roation=45)
plt.show()
```

### Ex. Regularized Regression: Ridge
```python
# Import Ridge

from sklearn.linear_model import Ridge

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

ridge_scores = []

for alpha in alphas:

  # Create a Ridge regression model

  ridge = Ridge(alpha = alpha)

  # Fit the data

  ridge.fit(X_train, y_train)

  # Obtain R-squared

  score = ridge.score(X_test, y_test)

  ridge_scores.append(score)

print(ridge_scores)
```
### Ex. Regularized Regression: Lasso
```python
# Import Lasso

from sklearn.linear_model import Lasso

  

# Instantiate a lasso regression model

lasso = Lasso(alpha=0.3)

  

# Fit the model to the data

lasso.fit(X,y)

  

# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)

plt.bar(sales_columns, lasso_coef)

plt.xticks(rotation=45)

plt.show()
```

# 3. Fine-tuning Your Model

| P/A   | Legit | Fraud |
| ----- | ----- | ----- |
| Legit | TN    | FP    |
| Fraud | FN    | TP    |

- Precision
	- TP / (TP + FP)
	- High precision, lower false positive rate
	- high precision: not many legitimate transactions are predicted to be fraudulent
- Recall/Sensitivity
	- TP / (TP + FN)
	- High recall: lower false negative
	- predicted most fraudulent transaction correctly
- F1 Score
	- 2  (Precision * recall )/ precision + recall
## Confusion matrix in scikit-learn
```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors = 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Logistic Regression and the ROC curve
- logistic regression is used for classification problems
- outputs probabilities
- if probability, p > 0.5
	- data is labeled as 1
- if probability, p < 0.5
	- data is labeled as 0
- Produces linear decision boundary
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# predicting probabilities
y_pred_probs = logreg.predict_proba(X_test)[:,1]
print(y_pred_probs[0])
```

Probability thresholds
- by default, logistic regression threshold = 0.5
- Not specified to logistic regression
	- KNN classifiers also have thresholds
The ROC curve
- ROC (Receiver Operating Characteristic)
- visualize how different thresholds affect the TP and FP rates
- If p = 0
	- model predicts 1, for all values
		- it will correctly predict all positive values, and incorrectly predict all negative values
- if p = 1
	- model predicts  0, for all values
		- both TP and FP rates are 0
```python
#plotting ROC curve
from sklearn.metrics import roc_curve
fpr, tpr thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

#ROC AUC
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```
- if the ROC curve is above the dotted line, it indicates that the model performs better than randomly guessing the class of each observation
## Hyperparameter tuning
- ridge/lasso regression: choosing alpha
- KNN: choosing n_neighbors
- Hyperparameters: parameters we specify before fitting the model
### Grid search cross-validation
![[grid-search cv.png]]
```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits = 5, shuffle = True, random_state=42)
param_grid = {"alpha": np.arrange(0.001, 1, 10), # use 10 evenly spaced values from 0.001 to 1
			"solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchSV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```
- limitations
	- it does not scale well
	- 3-fold cv, 1 hyp, 10 total values = 30 fits
	- 10-fold cv, 3 hyp, 30 total values = 900 fits
### RandomizedSearchCV
- picks random hyperparameters rather than exhausting every param
- n_iter: optional, determines the number of hyperparameters that will be used
```python
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits = 5, shuffle = True, random_state=42)
param_grid = {"alpha": np.arrange(0.001, 1, 10),
			"solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter = 2)
ridge_cv.fit(X_train, y_train)

# evaluation on the test set
test_score = ridge_cv.score(X_test, y_test)
print(test_score)
```

# 4. Preprocessing and Pipelines
## Preprocessing data
- scikit-learn requirements
	- numeric data
	- no missing values
- dealing with categorical features
	- need to convert categorical features into numerical values
	- convert to binary features called dummy variables
		- 0: observation was NOT that category
		- 1: Observation was that category
- Dealinng with categorical features in Python
	- scikit-learn: `OneHotEncoder()`
	- pandas: `get_dummies()`
```python
# Encoding dummy variables
import pandas as pd
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df["genre"], drop_frist=True)
print(music_dummies.head())

# bring back binary features to original DF
music_dummies = pd.concat([music_df, music_dummies], axis=1)
music_dummies = music_dummies.drop("genre", axis=1)
```

```python
#Linear regression with dummy variables
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_mode import LinearRegression
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error") 
#negative MSE because cv from sklearn presumre higher score, the better. negative is set to counter-act it

print(np.sqrt(-linreg_cv))
```
### Handling Missing Data

```python
print(music_df.isna().sum().sort_values())
```
- Drop missing Data
```python
music_df = music_df.dropna(subset=["genres", "popularity", "loudness", "liveness", "tempo"])
print(music_df.isna().sum().sort_values())
```
- Imputing values
	- use subject-matter expertise to replace missing data with educated guesses
	- Common to use the mean
	- Can also use the median, or another value
	- For categorical values, we typically use the most frequent value - the mode
	- must split our data first, to avoid ==data leakage==
	```python
	from sklearn.impute import SimpleImputer
	X_cat = music_df["genre"].values.reshape(-1,1)
	X_num = music_df.drop(["genre", "popularity"], axis = 1).values
	y= music_df["popularity"].values
	X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size = 0.2, random_state=12)
	X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=12)

	imp_cat = SimpleImputer(strategy="most_frequent")
	X_train_cat = imp_cat.transform(X_test_cat)
	X_test_cat = imp_cat.transform(X_test_cat)

	#numeric data
	imp_num = SimpleImputer()
	X_train_num = imp_num.fit_transform(X_train_num)
	X_test_num = imp_num.transform(X_test_num)
	X_train = np.append(X_train_num, X_train_cat, axis=1)
	X_test = np.append(X_test_num, X_test_cat, axis=1)
	```
- Imputing within a pipeline
	```python
	from sklearn.pipeline import Pipeline
	music_df = music_df.dropna(subset=["genres", "popularity", "loudness", "liveness", "tempo"])
	music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)
	X = music_df.drop("genre", axis=1).values
	y = music_df["genre"].values

	steps = [("imputation", SimpleImputer()),
			("logistic_regression", LogisticRegression())]
			
	pipeline = Pipeline(steps)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	pipeline.fit(X_train, y_train)
	pipeline.predict(X_test)
	pipeline.score(X_test, y_test)
	```
### Centering and Scaling
- why scale our data?
	- we want features to be on a similar scale
	- normalize or standardize data
- how to scale
	- subtract the mean and divide by variance
		- all features are centered around zero and have a variance of one
		- this is called standardization
	- can also subtract the minimum and divide by the range
		- minimum zero and maximum 1
	- can also normalize so the data ranges from -1 to 1
```python
from sklearn.preprocessing import StandardScaler
X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```
- Scaling in a pipeline
```python
steps = [('scaler', StandardScaler()), 
		('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))
```
- CV and scaling in a pipeline
```python
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()),
		('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {"knn__n_neighbors": np.arange(1,50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


cv = GridSearchCV(pipeline, param_grid = parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

#checking model parameters
print(cv.best_score_)
print(cv.best_params_)

```

### Centering and Scaling for Regression
```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()),

         ("lasso", Lasso(alpha=0.5))]

  

# Instantiate the pipeline

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

  

# Calculate and print R-squared

print(pipeline.score(X_test, y_test))
```
### Centering and Scaling for classification
```python
# Build the steps

steps = [("scaler", StandardScaler()),

         ("logreg", LogisticRegression())]

pipeline = Pipeline(steps)

  

# Create the parameter space

parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

                                                    random_state=21)

  

# Instantiate the grid search object

cv = GridSearchCV(pipeline, param_grid=parameters)

  

# Fit to the training data

cv.fit(X_train, y_train)

print(cv.best_score_, "\n", cv.best_params_)
```

## Evaluating Multiple Models
- some guiding principles
	- Size of the dataset
		- fewer features - simpler model, faster training time
		- some models require large amounts of data to perform well
	- Interpretability
		- Some models are easier to explain, which can be important for stakeholders
		- linear regression has high interpretability, as we can understand the coefficients
	- Flexibility
		- may improve accuracy, by making fewer assumptions about data
		- KNN is a more flexible model, doesn't assume linear relation between features and target
- It's all in the metrics
	- Regression model performance:
		- RMSE
		- R-squared
	- Classification model performance
		- Accuracy
		- Confusion matrix
		- Precision, recall, F1-score
		- ROC AUC
	- Train several models and evaluate performance out of the box (w/ hyperparameter tuning)
- A note on scaling
	- Models affected by scaling:
		- KNN
		- Linear Regression (plus Ridge, Lasso)
		- Logistic Regression
		- Artificial Neural Network
	- Scale data before evaluation model
```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn. Linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X = music.drop("genre", axis=1).values
y = music["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = sclaer.transform(X_test)

models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(),
"Decision Tree": DecisionTreeClassifier()}
results = []
for model in models.values():
	kf = KFold(n_splits=6, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
	results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()

#Test set performance
for name, model in models.items():
	model.fit(X_train_scaled, y_train)
	test_score = model.score(X_test_scaled, y_test)
	print("{} Test Set Accuracy: {}".format(name, test_score))
```

### Visualizing  regression model performance
```python
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}

results = []

# Loop through the models' values
for model in models.values():

  kf = KFold(n_splits=6, random_state=42, shuffle=True)

  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

  # Append the results
  results.append(cv_scores)

  

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()
```

### Predicting on Test set
```python
# Import root_mean_squared_error

from sklearn.metrics import root_mean_squared_error
for name, model in models.items():
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)

  # Calculate the test_rmse
  test_rmse = root_mean_squared_error(y_test, y_pred)

  print("{} Test Set RMSE: {}".format(name, test_rmse))
```
### Visualizing classification model performance
```python
# Create models dictionary

models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}

results = []

  

# Loop through the models' values

for model in models.values():

  # Instantiate a KFold object

  kf = KFold(n_splits=6, random_state=12, shuffle=True)

  # Perform cross-validation

  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)

  results.append(cv_results)

plt.boxplot(results, labels=models.keys())

plt.show()
```

### Pipeline for predicting song popularity
```python
# Create steps

steps = [("imp_mean", SimpleImputer()), 

         ("scaler", StandardScaler()), 

         ("logreg", LogisticRegression())]

  

# Set up pipeline

pipeline = Pipeline(steps)

params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],

         "logreg__C": np.linspace(0.001, 1.0, 10)}

  

# Create the GridSearchCV object

tuning = GridSearchCV(pipeline, param_grid=params)

tuning.fit(X_train, y_train)

y_pred = tuning.predict(X_test)

  

# Compute and print performance

print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))
```