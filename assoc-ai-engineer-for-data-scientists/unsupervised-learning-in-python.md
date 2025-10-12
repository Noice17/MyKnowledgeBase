# Clustering for Dataset Exploration
##  Unsupervised Learning
- finds patterns in data
- supervised find patterns for a prediction task
- unsupervised, learning without labels
    - finds patterns in data
### k-means clustering
- finds clusters of samples
- number of clusters must be specified
- implemented in `sklearn`
```python
print(samples)

from sklearn.cluster import KMeans
model = KMeans(n_clusters =3)
model.fit(sample)

KMeans(n_clusters=3)

labels = model.predict(samples)
print(labels)
```
### Cluster labels for new samples
- new sampels can be  assigned to existing clusters
- k-means remembers the mean of each cluster (the "centroids")
### Scatter plots
- each point represent a sample
```python
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()
```

```python
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()
```

## Evaluating a clustering
- can check correspondence with
### Cross tabulation with pandas
```python
import pandas as pd
df = pd.DataFrame({'labels':labels, 'species':species})
print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)
```
### Measuring clustering quality
- using only samples and their cluster labels
- a good clustering has tight clusters
- samples in each cluster bunched together

### Inertia measures clustering quality
- measures how spread out the clusters are (*lower* is better)
- distance from each sample to centroid of its cluster
- k-means attempts to minimize the inertia when choosing clusters
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)
```
### The number of clusters
- clustering of the iris dataset with different number of clusters
- more clusters means lower inertia
- a good clusterring has tight clusters (so low intertia) BUT not too many clusters
- choose an 'elbow' in the inertia plot

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

```

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters = 3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

```
## Transforming features for better clustering
### Feature Variance
- variance of a feature measures spread of its values
### StandardScaler
- in kmeans: feature variance = feature influence
- transforms each feature to have mean 0 and variance 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)
```
- `fit()` / `transform()` with StandardScaler
- `fit()` / `predict()` with KMeans
### StandardScaler, then KMeans
- use pipelines

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters = 3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)

Pipeline(step=...)
labels = pipeline.predict(samples)
```
### sklearn preprocessing steps
- StandardScaler, MaxAbsScaler, Normalizer

```python
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


#<script.py> output:
    # species  Bream  Pike  Roach  Smelt
    # labels                            
    # 0            0     0      0     13
    # 1           33     0      1      0
    # 2            0    17      0      0
    # 3            1     0     19      1
```

### Using Normalizer
```python
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters = 10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

#     labels                           companies
# 41       0                       Philip Morris
# 1        1                                 AIG
# 13       1                   DuPont de Nemours
# 3        1                    American express
# 14       1                                Dell
# 23       1                                 IBM
# 20       1                          Home Depot
# 8        1                         Caterpillar
# 26       1                      JPMorgan Chase
# 31       1                           McDonalds
# 30       1                          MasterCard
# 59       1                               Yahoo
# 35       1                            Navistar
# 47       1                            Symantec
# 32       1                                  3M
# 16       1                   General Electrics
# 58       1                               Xerox
# 55       1                         Wells Fargo
# 38       2                               Pepsi
# 25       2                   Johnson & Johnson
# 27       2                      Kimberly-Clark
# 9        2                   Colgate-Palmolive
# 40       2                      Procter Gamble
# 56       2                            Wal-Mart
# 19       3                     GlaxoSmithKline
# 39       3                              Pfizer
# 5        4                     Bank of America
# 15       4                                Ford
# 48       4                              Toyota
# 21       4                               Honda
# 18       4                       Goldman Sachs
# 7        4                               Canon
# 45       4                                Sony
# 34       4                          Mitsubishi
# 29       5                     Lookheed Martin
# 4        5                              Boeing
# 54       5                            Walgreen
# 36       5                    Northrop Grumman
# 37       6                            Novartis
# 43       6                                 SAP
# 52       6                            Unilever
# 6        6            British American Tobacco
# 42       6                   Royal Dutch Shell
# 46       6                      Sanofi-Aventis
# 49       6                               Total
# 10       7                      ConocoPhillips
# 12       7                             Chevron
# 28       7                           Coca Cola
# 57       7                               Exxon
# 44       7                        Schlumberger
# 53       7                       Valero Energy
# 33       8                           Microsoft
# 22       8                                  HP
# 11       8                               Cisco
# 50       8  Taiwan Semiconductor Manufacturing
# 24       8                               Intel
# 51       8                   Texas instruments
# 17       9                     Google/Alphabet
# 0        9                               Apple
# 2        9                              Amazon
```

# Visualization with Hierarchical Clusteringand t-SNE
## Visualizing Hierarchies
- t-SNE: Creates a 2D map of a dataset
- Hierarchical clustering
### Hierarchy of groups
- groups of living things can forma hierarchy
- custers are contained in one another
### Hierarchical clustering
- every country begins in a separate cluster
- at each step, the two closest clusters are merged
- continue untill all countries in a single cluster
- this is 'agglomerative' hierarchical clusterings
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_font_size=6)
plt show()
```

## Cluster labels in hierarchical clustering
- cluster labels at any intermediate stage can be recovered for use
- i.e. setting the height
- height on dendrogram = distance between merging clusters
- height specifies max.distance between merging clusters
### Distance between clusters
- defined by `linkage methods`
- `complete` linkage: distance between clusters in max. distance between their samples
- specified via method parameter, eg linkage(samples, method = 'complete')
- different linkage method, different hierarchical clustering

### Extracting cluster labels
- use `fcluster()`
- return NumPy array of cluster labels
```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method ='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')
print(labels)
```

### Aligning cluster labels with country names
```python
import pandas as pd
pairs = pd.DataFrames({'labels': labels, 'countries': country_names})
print(pairs.sort_values('labels'))
```
- scipy cluster start at 1 not 0 like in scikit

## t-SNE for 2-dimensional maps
- t-distributed stochastic neighbor embedding
- maps samples to 2D space (or 3D)
- map approximately preserves nearness of samples
- great for inspecting datasets

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()
```
### t-SNE has only fit_transform()
- simultaneously fits the model and transform data
- has no separate fit() and transform() methods
### t-SNE learning rate
- try values between 50 - 200

# Decorrelating Your Data and Dimension Reduction
## Dimension Reduction
- more efficient storage and computation
- remove less-informative "noise" features
### Principal Component Analysis
- PCA
- fundamental dimension redution technique
- first step "decorrelation"
- second step, reduces dimension
### PCA alignes data with axes
- rotates data samples to be aligned with axes
- shifts data samples so they have mean 0
no information is lost
### fit/transform pattern
- PCA is scikit-learn component like KMeans or StandardScaler
- fit learns the transformation from given data
- transform() applies the learned transformation
- transform can be applied to new data

```python
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)

transformed = model.transform(samples)
```
### PCA features
- rows of transformed correspond to samples
- columns of transformed are the PCA features
### PCA features are not correlated
- features of dataset are often correlated
- PCA aligns the data with axes
- resulting PCA features are not linearly correlated

### Pearson correlation
- measures linear correlation of features
- values between -1 and 1
- value of 0 means no linear correlation

### Principal components
- principal components = directions of variance
- PCA aligns principal components with axes
`print(model.components_)`

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)
```

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

## Intrinsic dimension
- the number of features needed to approximate the dataset
- essential idea behind dimension reduction
- can be detected with PCA
- PCA identifies intrinsic dimension
- scatter plots work only if samples have 2 or 3 features
- PCA identifies dimension when samples have any number of features
- Intrinsic dimension = number of PCA features with significant variance
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA Feature')
plt.show()
```

```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```

```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

```
## Dimension Reduction
- represents same data, using less features
- PCA features are in decreasing order of variance
- assumes the low variance features are "noise"
- `PCA(n_components=2)`: keeps the first 2 PCA features
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)

transformed = pca.transform(samples)
print(transformed.shape)


import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()
```
- discard low variance PCA features
- assumes the high variance features are informative

## Work frequency arrays
- rows represent documents, columns represent words
- entries measure presence of each word in each document
- `Sparse`: most entries are zero
    - can use `scipy.sparse.csr_matrix` instead of NumPy array
- `csr_matrix` remembers only the non-zero entries

## TruncatedSVD and csr_matrix
- sklearen PCA does not support `csr_matrix`
- use sklearn `TruncatedSVD`
```python
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)
transformed = model.transform(documents)
```

```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

```

## tf-idf
```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names_out()

# Print words
print(words)
```
## Clustering Wikipedia
```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))


```
# Discovering Interpretable Features
## Non-negative matrix factorization (NMF)
- dimension reduction technique
- NMF models are interpretable
- however, all sample features must be non-negative (>=0)
- NMF espresses documents as combinations of topics(or themes)
- NMF expresses images as combinations of patterns

- follows fit() / transform() pattern
- must specify number of components: NMF(n_components=2)
- works with NumPy arrays and with csr_matrix (sparse arrays)

### word-frequency array
- measure presence of words in each document using "tf-idf"
    - tf = frequency of word in document
    - idf = reduces influence of frequent words
```python
from sklearn.decomposition import NMF
model = NMF(n_components=2)

model.fit(samples)
nmf_features = model.transform(samples)

print(model.components_)
```
- Dimension of components = dimension of samples
- entries are non-negative
- NMF feature values are non-negative
- can be used to reconstruct samples

### Reconstrcution
- multiply components by feature values, and add up
- can also be expressed as a product of matrices
- this is the "Matrix Factorization"

## NMF learns interpretable parts
print(nmf.components_.shape)
- for documents
    - nmf components represent topics
    - nmf features combine topics into documents
- for images, NMF components are parts of images

### Grayscale images
- measures pixel brightness
- represent with value 0-1 (O is black)

- graysclae images as flat arrays
    - enumerate entries
    - row by row
    - from left to right, top-bottom

### Encoding a collection of images
- of the  same size
- encode a 2D array
- each row, 1 images
- each column corresponds to a pixel
```python
print(sample)
bitmap = sample.reshape((2,3))
print(bitmap)

from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation = 'nearest')
plt.show()
```

### Show Grayscale image
```python
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
```

```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)

def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
```

### comparison with PCA
```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
    
```

## Building Recommender Systems using NMF
### Strategy
- apply nmf to the word-frequency array
- nmf feature values describe the topics
    - so similar documents have similar NMF feature values
### Apply NMF to the word-frequency array
```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)
```
### Cosine similarity
- uses the angle between the lines
- high values means more similar
- max value is 1 when angle is 0 degrees

```python
from sklearn.preprocessing import normalize

norm_features = normalize(nmf_features)

current_article= norm_features[23.:]
similarities = norm_features.dot(current_article)
```

```python
import pandas as pd

norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_articlr = df.loc['Doc bites man']
similarities = df.dot(current_article)

print(similarities.nlargest())
```

### Recommend musical artists
```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())


```