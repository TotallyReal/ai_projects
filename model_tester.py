
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from misc import printProgressBar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training the model
def train_model(model: nn.Module, data_loader: DataLoader, epochs=10, learning_rate=0.001):
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model.train()  # Set model to training mode

    for epoch in range(epochs):
        data_so_far = 0
        printProgressBar(iteration=data_so_far, total=60000)
        for X_train, Y_train in data_loader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            # Forward pass
            Y_pred = model(X_train)
            loss = loss_function(Y_pred, Y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_so_far += X_train.shape[0]
            printProgressBar(iteration=data_so_far, total=60000)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Testing the model
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    total_errors = 0
    total_samples = 0

    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)

            outputs = model(X_test)
            predicted = torch.argmax(outputs, dim=1)

            total_errors += (predicted != Y_test).sum().item()
            total_samples += len(Y_test)

        error_rate = total_errors/total_samples
        print(f"Number of errors: {total_errors}/{total_samples} ~ {error_rate}")

    return error_rate