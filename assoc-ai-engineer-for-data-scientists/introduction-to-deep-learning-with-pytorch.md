# Introduction to PyTorch, a Deep Learning Library
## Deep learning networks
- inspired by how the human brain learns
- neurons -> neural networks
- models require large amount of data
- at least 100k data points

## PyTorch tensors
- similar to array or matrix
- building block of neural networks


```python
import torch

my_list = [[1,2,3], [4,5,6]]
tensor = torch.tensor(my_list)
print(tensor)

# tensor shape
print(tensor.shape)

# tensor data type
print(tensor.dtype)

# operations
# addition/subtraction: must be compatible (should have the same shape)
# element-wise multiplication
```

## Neural Networks and layers
- layers
    - input
    - hidden
    - output
- fully connectioned network
    - input -> output
    - equivalent to linear model
```python
import torch.nn as nn

#input neurons = features
#output neurons = classes

input_tensor = torch.tensor(
    [[0.3471, 0.4567, -0.2356]]
)

linear_layer = nn.Linear(
    in_features = 3,
    out_features = 2
)

output = linear_layer(input_tensor)
print(output)


print(linear_layer.weights)
print(linear_layer.bias)
```
- bias is to account for baseline information

## Hidden Layers and parameters
### Stacking layers with nn.Sequential()
```python
model - nn.Sequential(
    nn.Linear(n_features, 8), #no. of input features
    nn.Linear(8,4)
    nn.Linear(4, n_classes) #no. of output classes
)
```

- layers within nn.Sequential() are hidden layers
- n_features and n_classes are defined by the dataset

```python
model = nn.Sequential(
    nn.Linear(10,18) #takes 10, output 18
    nn.Linear(18,20) #takes 18, output 20
    nn.Linear(20,5) #takes 20, output 5
)
```
- Fully connected when each neuron links to all neurons in the previous layer
- a neuron in a linear layer:
    - performs a linear operation using all neurons from the previous layer
    - has N + 1 parameters: N from inputs and 1 for the bias
- more hidden layers =  more parameters = hhigher model capacity

```python
model = nn.Sequential(
    nn.Linear(8,4), # first layer has 4 neurons, each ahs 8+1 parameters. 9 * 4 = 36 parameters
    nn.Linear(4,2) # 2nd layers has 2 neurons, each has 4+1 parameters. 5 * 2 = 10 parameters
    )
    # model has 36 + 10 = 46 learnable parameters
```
- `.numel()`: returns the number of elements in the tensor

```python
total = 0

for parameter in model.parameters():
    total += parameter.numel()
print(total)
```
# Neural Network Architecture and Hyperparameters
## Discovering Activation functions
- activation functions
    - adds non-linearity to the network
    - Sigmoid: for binary classification
    - Softmax: for multi-calss classification
- networks can learn more complex relationships with non-linearity
- pre-activation output passed to the activation function
### Sigmoid function
- take the pre-activation output and pass it to the sigmoid function
- obtain a value between 0 and 1
- if output is > 0.5, class label = 1
- if output is < 0.5, class label = 0 
```python
import torch
import torch.nn as nn

input_tensor = torch.tensor([[6]])
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)
print(output)

model = nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,1),
    nn.Sigmoid() #put in last
)
```
- sigmoid as last step in network of linear layers is equivalent to traditional logistic regression

### Softmax
- takes three-dimensional as input and outputs the same shape
- outputs a probability distribution
    - each element is a probability(it's bounded between 0 and 1)
    - the sum of the output vector is equal to 1
```python
import torch
import torch.nn as nn

input_tensor = torch.tensor(
    [[4.3, 6.1, 2.3]]
)

probabilities = nn.Softmax(dim=-1) #dim -1 indicates softmax is applied to the input tensor's last dimension
output_tensor = probabilities(input_tensor)
print(output_tensor)

```
## Running a forward pass
- generating predictions
- input data flows through layers
- calculations performed at each layer
- final layer generates outputs

- outputs produce based on weights and biases
- used for training and making predictions

- possible outputs
    - binary classification
    - multi-class classification
    - regressions

### Binary Classification
- output is between 0 and 1
### Multi-class classification: forward pass
- each row sums to one
- predicted label = class with highest probability

### Regresion

## Using loss functions to assess model predictions
- tells use how good our model is during training
- takes a model prediction y' and ground truth y
- correct prediciton = low loss, the goal is to minimize loss

### One-hot encoding concepts
- loss = F(y, y')
- y is single integer
- y' is a tensor (prediction before softmax)
    - y' is a tensor with N dimensions
- convert an integer y to a tensor of zeros and ones

```python
import torch.nn.functional as F

print(F.one_hot(torch.tensor(0), num_classes = 3))
```

### Cross entropy loss in PyTorch
```python
from torch.nn import CrossEntropyLoss

scores = torch.tensor([-5.2, 4.6, 0.8])
one_hot_target = torch.tensor([1,0,0])

criterion = CrossEntropyLoss()
print(criterion(scores.double(), one_hot_target.double()))
```
- scores: model predictions before the final softmax function
- one_hot_target: one hot encoded ground truth label
- loss: a single float

```python
y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([0, 1, 0])

# Create the one-hot encoded vector using PyTorch
one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes=3)

print("One-hot vector using NumPy:", one_hot_numpy)
print("One-hot vector using PyTorch:", one_hot_pytorch)
```
```python
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), num_classes=4)

# Create the cross entropy loss function
criterion = CrossEntropyLoss()

# Calculate the cross entropy loss
loss = criterion(scores.double(), one_hot_label.double())
print(loss)
```

## Using derivatives to update model parameters
- derivatice represents the slope of the curve
- steep slopes: large steps, derivative is high
- gentler slopes: small steps, derivative is low
- valley floor: flat, derivative is zero
- derivatives = gradients
    - gradients help minimize loss, tune layer weights and biases
### Backpropagation
- consider a network made of 3 layers
    - begin with loss gradients for L2
    - use L2 to compute L1 gradients
    - repeat for all layers (L1, L0)

```python
model = nn.Sequential(
        nn.Linear(16,8),
        nn.Linear(8,4),
        nn.Linear(4,2)
)
prediction = model(sample)

criterion = CrossEntropyLoss()
loss - criterion(prediction, target)
loss.backward()

# access each layer's gradients
model[0].weight.grad
model[0].bias.grad
model[1].weight.grad
model[1].bias.grad
model[2].weight.grad
model[2].bias.grad
```
- updating model parameters manually
```python
lr = 0.001
weight = model[0].weight
weight_grad = model[0].weight.grad

weight = weight - lr * weight_grad


bias = model[0].bias
bias_grad = model[0].bias.grad

bias = bias - lr * bias_grad
```

### Gradient descent
- for non-convex functions, we will use gradient descent
- PyTorch simplifies this with optimizers
    - Stochastic gradient descent (SGD)

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.step()
```

# Training a Neural Network with PyTorch
## A deeper dive into loading data
```python
import numpy as np

features = animals.iloc[:,1:-1]
X = features.to_numpy()


target = animals.iloc[:,-1]
y = target.to_numpy()


#TensorDataset
import torch
from torch.utils.data import TensorDataset

dataset = TensorDataset(torch.tesnor(X), torch.tensor(y))

input_sample, label_sample = dataset[0]


# Data Loader
from torch.utils.data import DataLoader

batch_size = 2
shuffle = True


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#iterate over the data loader

for batch_inputs, batch_labels in dataloader:
    print('batch_inputs:', batch_inputs)
    print('batch_labels:', batch_labels)
```
- Epoch: one full pass through the training dataloader
- Generalization: model performs well with unseen data

## Writing our first training loop
1. Create a model
2. Choose a loss function
3. Define a dataset
4. Set an optimizer
5. Run a training loop
    - Calculate loss (forward pass)
    - Compute gradients (backpropagation)
    - Updating model parameters
### Mean Squared Error Loss
- MSE loss is the mean of the squared difference between predictions and ground truth
```python
def mean_squared_loss(prediction, target):
    return np.mean((prediction-target) **2)
```

- in PyTorch
```python
criterion = nn.MSELoss()
loss = criterion(prediction, target)
```

```python
dataset = TensorDatset(
    torch.tensor(features).float(),
    torch.tensor(target).float()
)

dataloader = DataLoader(dataset, batch_size = 4, shuffle=True)

model = nn.Sequential(
    nn.Linear(4,2),
    nn.Linear(2,1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# training loop

for epoch in range(num_epochs):
    for data in dataloader:
        optimizer.zero_grad() # set the gradients to zero
        
        # run a forward pass
        feature, target = data
        pred = model(feature)

        loss = criterion(pred, target) # compute loss
        loss.backward() # compute gradients

        optimizer.step() #update model's parameters
```
### Using MSELoss
```python
y_pred = np.array([3, 5.0, 2.5, 7.0])  
y = np.array([3.0, 4.5, 2.0, 8.0])     

# Calculate MSE using NumPy
mse_numpy = np.mean((y_pred-y)**2)

# Create the MSELoss function in PyTorch
criterion = nn.MSELoss()

# Calculate MSE using PyTorch
mse_pytorch = criterion(torch.tensor(y_pred), torch.tensor(y))

print("MSE (NumPy):", mse_numpy)
print("MSE (PyTorch):", mse_pytorch)
```
## ReLU activation Functions
- Limitations of the sigmoid and softmax functions
    - sigmoind
        - outputs bounded between 0 and 1
        - usable anywhere in a network
        - gradients:
            - very small for large and small values of x
            - cause saturation, leading to the vanishing gradients problem
    - softmax function also suffers from saturation
    - essentially, they are not ideal to be used in hidden layers, and best use in the last layer only
- Rectified Linear Unit (ReLU)
    - fx = max(x, 0)
    - for positive inputs: output equals input
    - for negative inputs: output is 0
    - has no upper bound, gradients do not approach 0
        - therefore, overcome vanishing gradients
```python
rely = nn.ReLU()
```
- Leay ReLU
    - positive inputs behave liek ReLU
    - negative inputs are scaled by a small coefficient ( default: 0.01)
    - gradients for negative inputs are non-zero
```python
leaky_relu = nn.LeakyReLU(
    negative_slope = 0.05
)
```

### Using ReLU and Leaky ReLU
```python
# Create a ReLU function with PyTorch
relu_pytorch = nn.ReLU()

x_pos = torch.tensor(2.0)
x_neg = torch.tensor(-3.0)

# Apply the ReLU function to the tensors
output_pos = relu_pytorch(x_pos)
output_neg = relu_pytorch(x_neg)

print("ReLU applied to positive value:", output_pos)
print("ReLU applied to negative value:", output_neg)
```

```python
# Create a leaky relu function in PyTorch
leaky_relu_pytorch = nn.LeakyReLU(
    negative_slope=0.05
)

x = torch.tensor(-2.0)
# Call the above function on the tensor x
output = leaky_relu_pytorch(x)
print(output)
```
## Learning rate and momentum
- training a neural network = solving an optimization problem
- Stochastic Gradient Descent (SGD) optimizer
```python
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
```
- `learning rate`: controls step size
    - too high: poor performance
    - too low: slow training
    - range: 0.01 - 0.0001
- `momentum`: adds inertia to avoid getting stuck
    - helps escape local minimum
    - too small: optimizer gets stuck
    - 0.85 - 0.99
- Impact of the learning rate:
    - optimal learning rate: step size decreases near zero as the gradient gets smaller
    - small learning rate: optimizer will take much longer to find the minimum
    - high learning rate: optimizer cannot find the minimum


# Evaluating and Improving Models

## Layer initialization
- a layer weights are initialized to small values
- keeping both the input data and layer weights small ensures stable outputs
```python
import torch.nn as nn
layer = nn.Linear(64,128)
print(layer.weight.min(), layer.weight.max())
```
- the output of a neuron in a linear layer is a weighted sum of inputs from the previous layer

```python
import torch.nn as nn

layer = nn.Linear(64,128)
nn.init.uniform_(layer.weight) # uniform distribution 0-1

print(layer.weight.min(), layer.weight.max())
```

### Transfer learning
- reusing a model trained on a first task for a second similar task
    - trained a model on US data scientist salaries
    - use weights to train on European salaries
```python
import torch

layer = nn.Linear(64,128)
torch.save(layer, 'layer.pth')

new_layer = torch.load('layer.pth')
```

### Fine-tuning
- a type of transfer learning
    - smaller learning rate
    - train part of the network (we freeze some of them)
    - TIP: freeze early layers of network and fine-tune layers closer to output layer
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64,128),
    nn.Linear(128, 256)
)

for name, param in model.named_parameters():
    if name - '0.weight':
        param.requires_grad = False # freeze 1st layer
```

### Sequence
1. Fine a model trained on a similar task
2. load pre-trained weights
3. freeze (or not) some of the layers in the model
4. train with a smaller learning rate
5. look at the loss values and see if the learning rate needs to be adjusted

### Freezing
```python
for name, param in model.named_parameters():
  
    # Check for first layer's weight
    if name == '0.weight':
   
        # Freeze this weight
        param.requires_grad = False
        
    # Check for second layer's weight
    if name == '1.weight':
      
        # Freeze this weight
        param.requires_grad = False
```

## Evaluating model performance
- training: 80-90
- validation: 10-20
- test: 5-10

### Calculating training loss
- for each epoch
    - sum the loss across all batches in the dataloader
    - compute the mean training loss at the end of the epoch
```python
training_loss = 0.0

for inputs, labels in trainloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    training_loss += loss.item()

epoch_loss = training_loss / len(trainloader)
```
### Calculating validation loss
```python
validation_loss = 0.0

model.eval()

with torch.no_grad(): #Disable gradients for efficiency
    for inputs, labels in validationloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()

epoch_loss = validation_loss / len(validationloader) # compute mean loss

model.train() # Switch back to training mode
```

### Calculating accuracy with torchmetrics
```python
import torchmetrics

metric = torchmetric.Accuracy(task = "multiclass", num_classes=3)

for features, labels in dataloader:
    outputs = model(features)
    metric.update(outputs, labels.argmax(dim=-1))

accuracy = metric.compute()
metric.reset()
```

### Fighting Overfitting
- overfitting: the model does not generalize to unseen data
    - model memorizes training data
    - performs well on training data but poorly on validation data
- possible cause
    - data is not large enough
        - solution: get more data / use data augmentation
    - model has too much capacity
        - solution: reduce model size / add dropout
    - weights are too large
        - solution: weight decay
- using dropout layer
    - randomly zeroes out elements of the input tensor during training
```python
model = nn.Sequential(
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Dropout(p=0.5)
)

features = torch.randn((1,8))
print(model(features))
```
#### dropout is added after the activation function
- behaves differently in training and evaluation
    - train: randomly deactivates neurons
    - evaluation: it is disabled, old neurons are activated
#### Regularization with weight decay
`optimizer = optim.SGD(model.parameter(), lr=0.001, weight_decay=0.0001)`
- controlled by the weight_decay parameter in the optimizer, typically set to a small value
- weight decause encourages smaller weights by adding a penalty during optimization
- helps reduce overfitting, keeping weights smaller and improving generalization

### Improving model performance
1. overfit the training set
    - modify the training loop to overfit a single data point
        - should reach 1.0 accuracy and 0 loss
    - sacle up to the entire training set
        - keep default hyperparameters
```python
features, labels = next(iter(dataloader))
for i in range(1000):
    outputs = model(features)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
2. reduce overfitting
- dropout
- data augmentation
- weight decay
- reducing model capacity

3. fine-tune hyperarameters
- grid search
    - test parameters at fixed intervals
```python
for factor in range(2,6):
    lr = 10 ** -factor
```
- random search
    - randomly selects values within range
    - typically more efficient than grid search
```python
factor = np.random.uniform(2,6)
lr = 10 ** -factor
```

    
