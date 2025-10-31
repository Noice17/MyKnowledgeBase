# Training Robust Neural Networks
## Pytorch and Object-oriented programming
```python
class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    def deposit(self, amount):
        self.balance += amount
```

```python
from torch.utils.data import Dataset

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label
```

```python
dataset_train = WaterDataset(
    "water_train.csv"
)

from torch.utils.data import DataLoader

dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
)

features, labels = next(iter(dataloader_train))
print(f"Features: {features}, \nLabels: {labels}")
```

- Sequential model definition
```python
net = nn.Sequential(
    nn.Linear(9,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1),
    nn.Sigmoid(),
)
```
- Class-based model definition
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
net = Net()

```

## Optimizers, training, and evaluation
- Training loop
```python
import torch.nn as nn
import torch.optim as optim

#Define loss function  and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

#Iterate over epochs and training batches
for epoch in range(1000):
    for features, labels in dataloader_train:
        # Clear gradients
        optimizer.zero_grad()
        #Forward pass: get model's outputs
        outputs = net(features)
        #Compute loss
        loss = criterion(
            outputs, labels.view(-1,1)
        )
        #Computer gradients
        loss.backward()
        #Optimizer's step: update params
        optimizer.step()
```
- SGD
    - update depends on learning rate
    - simple and efficient, for basic models
    - rarely used in practice
- Adaptive Gradient (Adagrad)
    - `optimizer = optim.Adagrad(net.parameters(), lr=0.01)`
    - adapts learning rate for each paramter
    - good for sparse data
    - may decrease the learning rate too fast
- Root Mean Square Propagation
    - `optimizer = optim.RMSprop(net.parameters(), lr=0.01)`
    - update for each parameter based on the size of its previous gradients
- Adaptive moment Estimation (Adam)
    - `optimizer = optim.Adam(net.parameters(), lr=0.001)`
    - the most versatile and widely used
    - RMSprop + gradient momentum
    - often used as the go-to-optimizer

### Model evalutation
```python
from torchmetrics import Accuracy

#Set up accuracy metric
acc = Accuracy(task="binary")

#put model in eval mode and iterate over test data batches with no gradients
net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        #Pass data to model to get predicted probabilities
        outputs = net(features)
        #Compute predicted labels
        preds = (outputs >= 0.5).float()
        #Update accuracy metric
        acc(preds, labels.view(-1,1))


accuracy = acc.compute()
print(f"Accuracy: {accuracy}")
```

## Vanishing and exploding gradients
- Vanishing gradients
    - gradients get smaller and smaller during backward pass
    - earlier layers get small parameter updates
    - model doesn't learn
- Exploding
    - gradients get bigger and bigger
    - parameter updates are too large
    - training diverges 
### Solution
- Proper weights initialization
    - variance of layer inputs = variance of layer outputs
    - variance of gradients the same before and after a layer
    - ReLU (and similar), we can use He/Kaiming initialization
```python
import torch.nn.init as init

init.kaiming_uniform(layer.weight)
print(layer.weight)
```
- He/Kaiming initialization
```python
init.kaiming_uniform_(self.fc1.weight)
init.kaiming_uniform_(self.fc2.weight)
init.kaiming_uniform_(
    self.fc3.weight,
    nonlinearity="sigmoid",
)
```
- in full model
```python
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module) :
def __ init __ (self):
    super() .__ init __ ()
    self.fc1 = nn.Linear(9, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 1)

    init.kaiming_uniform_(self.fc1.weight)
    init.kaiming_uniform_(self.fc2.weight)
    init.kaiming_uniform_(
        self.fc3.weight,
        nonlinearity="sigmoid",
    )

def forward(self, x):
    x = nn. functional.relu(self.fc1(x))
    x = nn. functional.relu(self.fc2(x))
    x = nn. functional.sigmoid(self.fc3(x))
    return x
```

### Activation functions
- ReLU
    - Often used as the default activation
    - `nn.functional.relu()`
- ELU
    - `nn.functiona.elu()`
    - non-zero gradients for negative values - helps against dying neurons
    - average  output around zero - helps agains vanishing gradients

### Batch normalization
After a layer:
1. Normalize the layer's outputs by:
    - Subtracting the mean
    - dividing by the standard deviation
2. Scale and shift normalized outputs using learned parameters
    - faster loss decrease
    - helps against unstable gradients

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9,16)
        self.bn1 = nn.BatchNorm1d(16)
        
        ...

    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)

```

- good activations
- batch normalization

# Images & Convolutional Neural Networks
```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

dataset_train = ImageFolder(
    "data/clouds_train",
    transform=train_transforms,
)

dataloader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1,
)

image, label = next(iter(dataloader_train))
print(image.shape)


image = image.squeeze().permute(1,2,0) # place channel dimension at the end
print(image.shape)

import matplotlin.pyplot as plt
plt.imshow(image)
plt.show()

# Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((128,128))
])

dataset_train = ImageFolder(
    "data/clouds/train",
    transform = train_transforms,
)

```
## Convolutional Neural Networks
- Linear Layer
    - why not use
        - slowe training
        - overfitting
        - dont recognize spatial patterns
- Convolutional layer
    - slide filters of parameters over the input
    - at each position, perform convolution
    - resulting feature map:
        - preserves spatial patterns from input
        - uses fewer parameters than linear layer
    - one filter = one feature map
    - apply activations to feature maps
    - all feature maps combined from the output
    - `nn.Conv2d(3, 32, kernel_size=3)`
- Convolution
    - compute dot product of input patch and filter
    - sum the result
- Zero padding
    - add a frame of zeros to convolutional layer's input
    `nn.Conv2d(3,32, kernel_size=3, padding=1)`
    - maintains spatial dimensions of the input and output tensors
    - ensures border pixels are treated equally to others
- Max pooling
    - slide non-overlapping window over input
    - at each position, retain only the maximum value
    - used after convolutional layers to reduce spatial dimension
    - `nn.MaxPool2d(kernel_size=2)`
### Convolutional Neural Network
```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64*16*16, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
```
- `feature_extractor`: (convolution, activation, pooling), repeated twice and flattened
- `classifer`: single linear layer
- `forward()`: pass input image through feature extractor and classifier

## Training Image Classifiers
```python
train_transforms transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor()
])
```

### Cross-Entropy loss
- binary class: binary cross-entropy(BCE) loss
- multi-class classification: cross-entropy loss
- `criterion = nn.CrossEntropyLoss()`

### Image classifier training loop
```python
net = Net(num_classes=7)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
## Evaluating image classifier
- Averaging multi-class metrics
```python
from torchmetrics import Recall

recall_per_class = Recall(task="multiclass", num_classes=7, average=None)

# Imbalanced datasets
recall_micro = Recall(task="multiclass", num_classes=7,average="micro")

# Care about performance on small classes
recall_macro = Recall(task="multiclass", num_classes=7, average="macro")

# consider errors in larger classes as more important
recall_weighted = Recall(task="multiclass", num_classes=7, average="weighted")
```
### Evaluation loop
```python
from torchmetrics import Precision, Recall

metrics_precision = Precision(
    task="multiclass", num_classes=7, average="macro"
)

metrics_recall = Recall(
    task="multiclass", num_classes=7, average="macro"
)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs,1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)
precison = metric_precision.compute()
recall = metric_recall.compute()
```

```python

# Define precision metric
metric_precision = Precision(
    task="multiclass", num_classes=7, average=None
)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
precision = metric_precision.compute()

# Get precision per class
precision_per_class = {
    k: precision[v].item()
    for k, v 
    in dataset_test.class_to_idx.items()
}
print(precision_per_class)
```
# Sequences & Recurrent Neural Networks
## Sequential data
- ordered in time or space
- order of the data points contains dependencies between them

- NO RANDOM SPLITTING FOR TIME SERIES DATA
    - Look-ahed bias: model has info about the future
- split by time instead
    - split first 3 yrs, test in last year
- Creating sequences
    - sequence length = number of data points in one training sample
```python
import numpy as np

# Take data and sequence length as inputs
def create_sequences(df, seq_length):
    # Initialize inputs and targets lists
    xs, ys = [], []

    # Iterate over data points
    for i in range(len(df) - seq_length):
        #define inputs and target
        x = df.iloc[i:(i+seq_length), 1]
        y = df.iloc[i+seq_length, 1]
        # Append to pre-initialized lists
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Tensor Dataset
X_train, y_train = create_sequences(train_data, seq_length)
print(X_train.shape, y_train.shape)

## Convert to a torch dataset
from torch.utils data import TensorDataset

dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float(),
)

```
## Recurrent Neural Networks
- Recurrent neuron
    - feed-forward networks
    - RNN: have connectiosn pointing back
    - Recurrent neuron:
        - Input x
        - Output y
        - Hidden state h
- Sequence-to-sequence architecture
    - pass sequence as input, use the entire output sequence
    - eg. Real-time  speech recognition
- Sequence-to-vector architecture
    - pass sequence as input, use only the last output
    - eg. text topic classification, sentiment classification, time-series prediction one step ahead
- Vector-to-sequence architecture
    - pass single input, use the entire output sequence
    - eg. Text generation, Image captioning
- Encoder-decoder architecture
    - pass entire inut sequence, only then start using output sequence
    - eg. Non-real-time speech recognition: machine listens before it replies, Machine translation of text
```python
# RNN in PyTorch
class Net(nn.Module):
    # Define model class __init__ method
    def __init__(self):
        super().__init__()

        # Define recurrent layer
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

        # Define linear layer
        self.fc = nn.Linear(32,1)
    
    # Initialize first hidden state to zeros in forward()
    def forward(self,x):
        h0 = torch.zeros(2, x.size(0), 32)

        # Pass input and first hidden state through RNN layer
        out, _ = self.rnn(x, h0)

        # Select last RNN's output and pass it through linear layer
        out = self.fc(out[:, -1, :])
        return out

```
## LSTM and GRU cells
- Short-term memory problem
    - RNN cells maintain memory via hidden state
    - this memory is very short-term
- RNN cell
    - two inputs
        - current input data `x`
        - previous hidden state `h`
    - two outputs
        - current output `y`
        - next hidden state `h`
- LSTM cell
    - three inputs and outputs (two hidden states):
        - `h`: short-term state
        - `c`: long-term state
    - three gates:
        - Forget gate: what to remove from long-term memory
        - Input gate: what to save to long-term memory
        - Output gate: what to return at the current time step
```python
# LSTM in pytorch

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm - nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32,1)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        c0 = torch.zeros(2, x.size(0), 32)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```
- GRU cell
    - simplified version of LST cell
    - just one hidden state
    - no output gate
```python
# GRU in pytorch

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru - nn.GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32,1)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        out, _ = self.gru(x,h0)
        out = self.fc(out[:, -1, :])
        return out
```

## Training and evaluating RNNs
- Mean Squared Error Loss
    - Error: prediction - target
    - Squared error: (p - t ) **2
    - mean squared error
    - `criterion = nn.MSELoss()`
- Expanding tensors
    - recurrent layers expect input shape
        - `(batch_size, seq_length, num_features)`
    - but we got
        - `(batch_size, seq_length)`
    - we must add one dimension at the nd
        - `seqs = seqs.view(32, 96, 1)`
- Squeezing tensors
    - in evaluation loop, we need to revert the reshaping done ine the training loop
    - labels are of shape `(batch_size)`
    - Model outputs are `(batch_size, 1)`
    - shapes of model outputs and labels must match for the loss function
    - we can drop the last dimension from model outputs
        - `out = net(seqs).squeeze()`
```python
# Training Loop
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    net.parameters(), lr=0.001
)

for epoch in range(num_epochs):
    for seqs, labels in dataloader_train:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaludation loop

mse = torchmetrics.MeanSquaredError()

net.eval()
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs).squeeze()
        mse(outputs, labels)
mse.compute()
```

# Multi-Input & Multi-Output Architectures
- Multi-input models
    - using more information
    - multi-model models
    - metric learning
    - self-supervised learning
## Two-input Dataset
```python
from PIL import Image

class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
        self.transform = transform
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img, alphabet, label
```

## Tensor concatenation
```python
x = torch.tensor([[1,2,3]])
y = torch.tensor([[4,5,6]])

torch.cat((x,y), dim=0) #horizontal
torch.cat((x,y), dim=1) #vertical
```

## Two-input architecture
```python
class Net(nn.Module):
    def __init__(self):

        #image processing layer
        super().__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32,128)
        )

        # alphabet processing layer
        self.alphabet_layer = nn.Sequential(
            nn.Linear(30,8),
            nn.ELU(),
        )
        
        # classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 964)
        )
    def forward(self, x_image, x_alphabet):
        x_image = self.image_layer(x_image)
        x_alphabet = self.alphabet_layer(x_alphabet)
        x = torch.cat((x_image, x_alphabet), dim=1)
        return self.classifier(x)

## Training loop
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(10):
    for img, alpha, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(img, alpha)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Multi-output models
- multi-task learning
- multi-label classification
- regularization
```python
class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
        self.transform = transform
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img, alphabet, label
```
### Two-output architecture
```python
class Net(nn.Module):
    def __ init __ (self. num_alpha, num_char):
        super() .__ init __ ()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn. Flatten(),
            nn.Linear(16*32*32, 128)
        )

        self.classifier_alpha = nn.Linear(128, 30)
        self.classifier_char = nn.Linear(128, 964)

    def forward(self, x):
        x_image = self.image_layer(x)
        output_alpha = self.classifier_alpha(x_image)
        output_char = self.classifier_char(x_image)
        return output_alpha, output_char
```
### Training loop
```python
for each in range(10):
    for images, labels_alpha, labels_char in dataloader_train:
        optimizer.zero_grad()
        outputs_alpha, outputs_char = net(images)
        loss_alpha = criterion(
            outputs_alpha, labels_alpha
        )
        loss_char = criterion(
            outputs_char, labels_char
        )
        loss = loss_alpha + loss_char
        loss.backward()
        optimizer.step()
```

```python
# Print the sample at index 100
print(samples[100])

# Create dataset_train
dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
      	transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

# Create dataloader_train
dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=32,
)
```
## Evaluation of multi-output models and loss weighting
- Model Evaluation
```python
acc_alpha = Accuracy(
    task="multiclass", num_classes=30
)
acc_char = Accuracy(
    task="multiclass", num_classes=964
)

net.eval()
with torch.no_grad():
    for images, labels_alpha, labels_char in dataloader_test:
        out_alpha, out_char = net(images)
        _, pred_alpha = torc.max(out_alpha, 1)
        _, pred_char = torch.max(out_char, 1)
        acc_alpha(pred_alpha, labels_alpha)
        acc_char(pred_char, labels_char)

acc_alpha.compute()
acc_char.compute()
```
-