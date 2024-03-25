import numpy as np
import matplotlib.pyplot as plt
# setting seed
seed_value = 42
np.random.seed(42)

"""###Task 0: Load,Visualize & Preprocess Dataset"""
from keras import datasets
# load the dataset using keras library
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# visualizing the data using matplotlib
dim = x_train.shape[1]
digits = np.unique(y_train)
fig, axs = plt.subplots(10, 10, figsize=(12,12))

for i,d in enumerate(digits):
    for j in range(10):
        axs[i,j].imshow(x_train[y_train==d][j].reshape((dim,dim)), cmap='gray',interpolation='none')
        axs[i,j].axis('off')

# concatenating the train and test set into one ndarray
X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

X = X.reshape(X.shape[0],-1) # flattening the 2darray
X = X / 255.0  # normalizing pixel values

# merging all data into a single array dims = (number_of_images, 28*28+1)
mnist_data = np.column_stack((X,y))
print(mnist_data.shape, type(mnist_data))

"""###Task 1: Function for Image Rotation"""

def rotate(image, angle):
    """
    Rotate a 28x28 image array by the specified angle.

    Parameters:
    - image (numpy.ndarray): input image array of shape (28, 28).
    - angle (float): rotation angle in degrees.

    Returns:
    - numpy.ndarray: rotated image array of the same shape.
    """
    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.from_numpy(image).float()

    # Move tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    rad = torch.radians(torch.tensor(angle))

    # Rotation matrix
    rotation_matrix = torch.tensor([[torch.cos(rad), -torch.sin(rad)],
                                    [torch.sin(rad), torch.cos(rad)]], device=image_tensor.device)

    # Image center coordinates
    center = torch.tensor(image.shape) / 2.0

    # Apply rotation
    rotated_image = torch.zeros_like(image_tensor)
    for i in range(image_tensor.shape[0]):
        for j in range(image_tensor.shape[1]):
            # Translate coordinates to center
            translated_coords = torch.tensor([i, j], device=image_tensor.device) - center

            # Apply rotation matrix
            rotated_coords = torch.matmul(rotation_matrix, translated_coords)

            # Translate coordinates back to the original position
            new_coords = rotated_coords + center

            # Interpolate pixel values
            x, y = new_coords.int()
            if 0 <= x < image_tensor.shape[0] and 0 <= y < image_tensor.shape[1]:
                rotated_image[i, j] = image_tensor[x, y]

    # Move tensor back to CPU and convert to NumPy array
    rotated_image = rotated_image.cpu().numpy()

    return rotated_image

"""### Task 2: Generate Oversampled Dataset"""

def generate_dataset(dataset, oversample_rate = 0.2):
    to_generate_count = int(dataset.shape[0] * oversample_rate)

    updated_dataset = np.copy(dataset)

    for _ in range(to_generate_count):
        # randomly pick a data point from the original dataset
        idx = np.random.randint(0, dataset.shape[0])
        original_image = dataset[idx, :-1]
        label = dataset[idx, -1]

        # randomly choose an angle from the specified set of angles
        angles = [-30, -20, -10, 10, 20, 30]
        random_angle = np.random.choice(angles)

        # rotate the data point
        original_image = original_image.reshape(28,28)
        rotated_image = rotate(original_image, random_angle).reshape(-1)
        rotated_image = np.append(rotated_image, label)
        # add the new image to the dataset
        updated_dataset = np.vstack((updated_dataset, rotated_image))
    return updated_dataset

"""### The Model Building"""

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # settings the gradients to zero
        output = model(data)  # forward pass
        loss = criterion(output, target) # compute loss
        loss.backward() # backpropagation
        optimizer.step() # update parameters
        total_loss += loss.item() # compute total loss

    return total_loss / len(train_loader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            # append original labels and predicted labels to the lists
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions

def calculate_accuracy(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    # ensure both arrays have the same length
    assert len(labels) == len(predictions), "Arrays must have the same length"

    correct_predictions = np.sum(labels == predictions)
    total_predictions = len(labels)
    return correct_predictions / total_predictions

"""### Task 3: Wrapper Function to Learn Model"""

def learn_model(dataset):
    # extract features and labels from the dataset
    pixels = dataset[:, :-1].reshape(-1,1,28,28)
    labels = dataset[:, -1]
    batch_size = 64

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(pixels, labels, test_size=0.3, random_state=seed_value)

    # convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    # create DataLoader for training and validation sets
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # initialize CNN model
    model = SimpleCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        val_labels, val_preds = evaluate_model(model, val_loader, device)

        # assuming we have a function to calculate accuracy
        val_accuracy = calculate_accuracy(val_labels, val_preds)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

    # plot the training loss vs epochs
    plt.figure()
    plt.plot(np.arange(num_epochs), train_losses, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.show()

    return model

"""### Task 4: Monitor Performance"""

def monitor_perf(model, ground_truth_dataset, threshold):
    # convert ground truth data to PyTorch tensors
    x_truth = torch.tensor(ground_truth_dataset[:,:-1].reshape(-1,1,28,28), dtype=torch.float32)
    y_truth = torch.tensor(ground_truth_dataset[:,-1], dtype=torch.long)

    # create DataLoader for the ground truth set
    ground_truth_loader = DataLoader(TensorDataset(x_truth, y_truth), batch_size=1000, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_labels, all_predictions = evaluate_model(model, ground_truth_loader, device)
    accuracy = calculate_accuracy(all_labels, all_predictions)
    print()

    print(f"Monitoring Accuracy: {accuracy:.4f}")

    drift_flag = (1.0 - accuracy) > threshold

    if drift_flag:
        print("Drift Detected!")

    return drift_flag

"""### End-to-End Evaluation"""

# seperating the ground_truth data and usable data
features = mnist_data[:, :-1]
labels = mnist_data[:, -1]
x_train_data, x_ground_truth, y_train_data, y_ground_truth = train_test_split(features, labels, test_size=0.2, random_state=seed_value)

ground_truth = np.column_stack((x_ground_truth,y_ground_truth))
train_data = np.column_stack((x_train_data, y_train_data))
print(ground_truth.shape, train_data.shape)

sample_size = 1000
threshold = 0.02
sampling_rate = 0.2

# generating rotated samples initially
train_data = generate_dataset(train_data, sampling_rate)
ground_truth = generate_dataset(ground_truth, sampling_rate)

cnn_model = learn_model(train_data[:sample_size,:])
drift = monitor_perf(cnn_model, ground_truth, threshold)
while drift:
  print("------------------------RETRAINING MODEL------------------------")
  sample_size = sample_size + 1000
  if sample_size > len(train_data):
    train_data = generate_dataset(train_data, sampling_rate)
    ground_truth = generate_dataset(ground_truth, sampling_rate)
    sampling_rate += 0.2

  cnn_model = learn_model(train_data[:sample_size,:])
  drift = drift = monitor_perf(cnn_model, ground_truth, threshold)
