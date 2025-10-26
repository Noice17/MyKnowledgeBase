# Foundations of Explainable AI
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)


from sklearn.tree export_text
rules = export_text(model, feature_names=list(X_train.columns))
```
```python
model = DecisionTreeClassifier(random_state=42, max_depth=2)
model.fit(X_train, y_train)

# Extract the rules
rules = export_text(model, feature_names=list(X_train.columns))
print(rules)

y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

```python
from sklearn.tree import MLPClassifer

model = MLPClassifer(hidden_layer_sizes=(1000,1000))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
```
- model-agnostic approaches can be applied to decision trees
- model-agnostic approaches can be applies to neural networks

## Explainability in linear models
- Linear regression
    - predicts continuous values
- logistic regression
    - used for binary classification
- coefficients
    - tells us importance of each features
        - higher absolute value -> higher importance
        - lower absolute value -> lower importance
    - to compare coefficients -> absolute values
    - note: normalize feature scales before computing coefficients
```python
X_train = data.drop(['Chance of Admit', 'Accept'], axis=1)
y_reg = data['Chance of Admit']
y_cls = data['Accept']

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_cls)

print(lin_reg.coef_)
print(log_reg.coef_)
```

```python
# Standardize the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()

# Fit the model
model.fit(X_train_scaled, y_train)

# Derive coefficients
coefficients = model.coef_
feature_names = X_train.columns

# Plot coefficients
plt.bar(feature_names, coefficients)
plt.show()
```

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Derive coefficients
coefficients = model.coef_[0]
feature_names = X_train.columns

# Plot coefficients
plt.bar(feature_names, coefficients)
plt.show()

```
## Explainability in tree-based models
- Decision tree
    - fundamenta block of tree-based models
    - used for regression and classification
    - tree-like structure for predictions
        - several decisions
        - each decision is based on one feature
    - inheretly explainable
- Random forest
    - consits of many decision trees
    - used for regerssion and classification
    - complicates direct interpretability
    - feature importance
        - measures reduction of uncertainty in predictions
        - different than coefficients in linear models
```python
X_train = data.drop(['Accept'], axis=1)
y_train = data['Accept']


# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
print(tree_model.feature_importances_)

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)

print(forest_model.feature_importances_)

# feature importance
import matplotlib.pyplot as plt
plt.barh(X_train.columns, tree_model.feature_importances_)
#plt.bar(X_train.columns, forest_model.feature_importances_)
plt.title('Feature Importance')
plt.show()
```
### Feature importances

# Model-Agnostic Explainability

## Permutation Importance
- assesses feature importance
    - measures effect of feature shuffling on performance
- high versatile
```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10,10))
model.fit(X_train, y_train)


from sklearn.inspection import permutation_importance
result = permutation_importance(
    model,
    X_train, y_train,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

print(result.importances_mean)

import matplotlib.pyplot as plt
plt.bar(X_train.columns, result.importances_mean)
```
- comparing with logistic reg coefficients
```python
from sklearn.inspection import permutation_importance

# Extract and store model coefficients
coefficients = model.coef_[0]

# Compute permutation importance on the test set
perm_importance = permutation_importance(
    model,
    X,y,
    n_repeats=20,
    random_state=1,
    scoring='accuracy'
)

# Compute the average permutation importance
avg_perm_importance = perm_importance.importances_mean

plot_importances(coefficients, avg_perm_importance)
```

## SHAP Explainability
- SHapley Additive exPlanations
- uses shapely values from game theory
- SHAP values -> quantify feature contributions to predictions
- Explainers 
    - General explainers: can be applied to any model
    - type-specific explainers: optimized for specific model types
        -  Tree explainers
            - tree-based models
```python
import shap

# REGRESSION
explainer_reg = shap.TreeExplainer(rf_reg)
shap_values_reg = explainer_reg.shap_values(X)

# CLASSIFICATION
explainer_class = shap.TreeExplainer(rf_class)
shap_values_class = explainer_class.shap_values(X)

print(shap_values_reg.shape) # same as dataset
print(shap_values_class.shape) # additional dimension to accommodate classes

# select values of positive class
positive_values = shap_values_class[:,:,1]


# FEATURE IMPORTANCE
mean_shap_values_class = np.abs(shap_values_class[:,:,1]).mean(axis=0)
mean_shap_values_reg = np.abs(shap_values_reg).mean(axis=0)

plt.bar(X_train.columns, mean_shap_values_class)
plt.bar(X_train.columns, mean_shap_values_reg)
```
```python
import shap

# Create a SHAP Tree Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()
```

```python
import shap

# Create a SHAP Tree Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values[:,:,1]).mean(axis=0)

plt.bar(X.columns, mean_abs_shap)
plt.title('Mean Absolute SHAP Values for RandomForest')
plt.xticks(rotation=45)
plt.show()
```

## SHAP kernel explainer
- General explainer
    - Derives SHAP valeus for any model
        - K-nearest neighbors
        - Neural networks
        - Tree-based models
- slower than type-specific explainers
- universally applicable
```python
# MLPRegressor
import shap

explainer = shap.KernelExplainer(
    # model's prediction function
    mlp_reg.predict,

    # representative summary of dataset
    shap.kmeans(X,10)
)

shap_values_reg = explainer.shap_values(X)
mean_reg = np.abs(shap_values_reg).mean(axis=0)
plt.bar(X.columns, mean_reg)
```
```python
# MLPClassifier
import shap

explainer = shap.KernelExplainer(
    # model's prediction function
    mlp_clf.predict_proba,

    # representative summary of dataset
    shap.kmeans(X,10)
)

shap_values_cls = explainer.shap_values(X)
mean_cls = np.abs(shap_values_cls[:,:,1]).mean(axis=0)
plt.bar(X.columns, mean_cls)
```
### Comparing with model-specific approaches
- when compared with Linear regression and Logistic regression, the essential features identified are the same
```python
plt.bar(X.columns, np.abs(lin_reg.coef_))
plt.bar(X.columns, np.abs(log_reg.coef_[0]))
```
- Comparing a Logistic Regression with SHAP
```python
import shap

# Extract model coefficients
coefficients = model.coef_[0]

# Compute SHAP values
explainer = shap.KernelExplainer(
    model.predict_proba,
    shap.kmeans(X,10)
)
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values[:,:,1]).mean(axis=0)

plot_importances(coefficients, mean_abs_shap)
```

## Visualizing SHAP explainability
```python
shap.summary_plot(shap_values, X, plot_type="bar")
```
### Beeswarm plot
    - shows SHAP values distribution
    - highlights direction and magnitude of each feature prediction
```python
shap.summary_plot(shap_values, X, plot_type="dot")
```

### Partial dependence plot
- shows relationshio between feature and predicted outcome
- shows feature's impact across its range
- verifies if relationship is expected
```python
shap.partial_dependence_plot("age", model.predict, X)
```

# Local Explainability
- explains prediction for specific data point
- crucial for sensitive application
```python
explainer = shap.KernelExplainer(knn.predict_proba, shap.kmeans(X,10))

test_instance = X.iloc[0,:]
shap_values = explainer.shap_values(test_instance)
print(shap_values.shape)
```
- Waterfall plots
    - shows how features increase or decrease model's prediction
```python
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[:,1],
        base_values=explainer.expected_value[1],
        data=test_instance,
        feature_names=X.columns

    )
)
```
## Local explainability with LIME
- LIME: Local Interpretable Model-Agnostic Explanations
- explains predictions of complex models
- LIME Explainers
    - Tabular Explainer: structured data
    - Text Explainer
    - Image Explainer
- generates perturbations around a sample
- sees effect on model's output
- contructs simpler model for explanation
### Tabular Explainer
```python
# Regression
from lime.lime_tabular import LimeTabularExplainer
instance = X.iloc[1,:]

explainer_reg = LimeTabularExplainer(
    X.values,
    feature_names = X.columns,
    mode='regression'
)

explanation_reg = explainer_reg.explain_instance(
    instance.values,
    regressor.predict
)

explanation_reg.as_pyplot_figure()
```

```python
# Classification
from lime.lime_tabular import LimeTabularExplainer
instance = X.iloc[1,:]

explainer_class = LimeTabularExplainer(
    X.values,
    feature_names = X.columns,
    mode='classification'
)

explanation_class = explainer_class.explain_instance(
    instance.values,
    classifier.predict_proba
)

explanation_class.as_pyplot_figure()
```
- in comparison with SHAP, LIME's local explanation show the specific conditions the and ranges that detail how features influence an individual prediction, enhancing transparency and trust in model predictiosn at a granular level

## Text and image explainability with LIME
### Text-based models
- LIME text explainer
```python
from lime.lime_text import LimeTextExplainer

text_instance = "This product has great features but a poor design"

def model_predict(instance):
    ...
    return class_proabilities

explainer = LimeTextExplainer()
exp = explainer.explain_instance(
    text_instance,
    model_predict
)

exp.as_pyplot_figure()
```

### Image-based models
- LimeImageExplainer
```python
from lime.lime_image import LimeImageExplainer

explainer = LimeImageExplainer()
explanation = explainer.explain_instance(
    image,
    model_predict,
    num_samples=50
)

temp, _ = explanation.get_image_and_mask(
    explanation.top_labels[0]
    hide_rest=True
)

plt.imshow(temp)

```
# Advanced topics in explainable AI
## Explainability Metrics
- Consistency
    - assesses stability of explanation when model trained on different subsets
    - low consistency -> no robust explanation
    - Cosine similarity to measure consistency
```python
from sklearn.metrics.pairwise import cosine_similarity

explainer1 = shap.TreeExplainer(model1)
explainer2 = shap.TreeExplainer(model2)

shap_values1 = explainer1.shap_values(X1)
shap_values2 = explainer2.shap_values(X2)

feature_importance1 = np.mean(np.abs(shap_values1), axis=0)
feature_importance2 = np.mean(np.abs(shap_values2), axis=0)

consistency = cosine_similarity([feature_importance1], [feature_importance2])
print(consistency)
```
- Faithfulness
    - evaluates if importance features influence model's predictions
    - low faithfulness -> misleads trust in model reasoning
    - useful in sensitive applications
```python
X_intance = X_test.iloc[0]
original_prediction = model.predict_proba(X_intance)[0,1]
print(f"Original prediction: {original_prediction}")

X_instance['GRE Score'] = 310

new_prediction = model.predict_proba(X_instance) [0, 1]
print(f"Prediction after perturbing {important_feature}: {new_prediction}")

faithfulness_score = np.abs(original_prediction - new_prediction)
print(f"Local Faithfulness Score: {faithfulness_score}")
```
## Explaining unsupervised models
### Silhouette score
- measures clustering quality
- ranges from -1 to 1
- Impact(f) > 0 -> positive contribution for f
- Impact(f) < 0 -> f introduces noise
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=2).fit(X)
original_score = silhouette_score(X, kmeans.labels_)

for i in range(X.shape[1]):
    X_reduced = np.delete(X, i, axis=1)
    kmeans.fit(X_reduced)
    new_score = silhouette_score(X_reduced, kmeans.labels_)
    impact = original_score - new_score
    print({column_names[i]} {impact})
```
### Adjusted rand index(ARI)
- measures how well cluster assignments match
- maximum ARI = 1 -> perfect cluster alignment
- Lower ARI -> greater difference in clusterings
```python
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=2).fit(X)
original_clusters = kmeans.predict(X)

for i in range(X.shape[1]):
    X_reduced = np.delete(X, i, axis=1)
    reduced_clusters = kmeans.fit_predict(X_reduced)
    importance = 1 - adjusted_rand_score(original_clusters, reduced_clusters)
    print({df.columns[i]}{importance})
```

## Explaining chat-based generative AI models
- Chain-of-thought prompt
    - encourages model to articulate its reasoning
```python
# Complete the chain-of-thought prompt
prompt = "In a basket, there are twice as many apples as oranges and three times as many oranges as bananas. If there are 6 bananas in the basket, how many fruits are there in total? Show you reasoning step-by-step"

response = get_response(prompt)
print(response)
```
- self-consistency
    - useful for text classification tasks
```python
prompt = """Classify the following review as positive or negative.
You should reply with either "positive" or "negative", nothing else.
Review: 'The customer service was great, but the product itself did not meet my expectations. '"""

responses = []
for i in range(5):
sentiment = get_response (review)
responses.append(sentiment.lower())

confidence = {
    'positive': responses.count ('positive') / len(responses),
    'negative': responses.count ('negative') / len(responses)
}

```