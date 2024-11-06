import time
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lp_optim import LPAnnealingAdam, LPAnnealingAGD

# Define a simple neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)    

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def l_0_norm(model: nn.Module):
    return ([torch.sum(torch.abs(param))  for param in model.parameters()])

# Load the MNIST dataset
def load_data(batch_size, dataset_name='MNIST'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        dataset_class = getattr(datasets, dataset_name)
    except AttributeError:
        raise ValueError(f"Dataset {dataset_name} is not available in torchvision.datasets.")


    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_model_complexity(model):
    l0 = sum([(param != 0.0).sum().item() for param in model.parameters()])
    num_params = sum([param.numel() for param in model.parameters()])
    return l0, num_params

def format_time(seconds):
    """
    Converts a time duration from seconds into a human-readable format (hh:mm:ss).

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Time duration formatted as hh:mm:ss.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:02}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)

def log_message(message, log_path=None):
    """
    Logs a message to the console and optionally writes it to a file.

    Args:
        message (str): The message to log.
        file_path (str, optional): Path to the file where the message should be written. Defaults to None.
    """
    print(message)  # Print to console
    if log_path:
        with open(log_path, 'a') as file:
            file.write(message + '\n')  # Write to file

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy

def validate(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        l0, num_params = get_model_complexity(model)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    message = (
        f'\nTest set: '
        f'Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), '
        f'Model complexity: {l0}/{num_params} parameters\n'
    )
    log_message(message)
    return test_loss, accuracy

def perform_baseline(model, device, num_epochs, train_loader, test_loader, criterion, log_path):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        train_accuracy = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_accuracy = validate(model, device, test_loader, criterion, epoch)
        l0, num_params = get_model_complexity(model)
        log_message(
            f'Epoch {epoch + 1}/{num_epochs}, '
            f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, '
            f'Model Complexity: {l0}/{num_params}\n',
            log_path)
    
    # Write the results of the final epoch to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["BaselineAdam", "NA", "NA", l0, test_accuracy, test_loss])

def perform_grid_search(gamma_values, decay_rate_values, num_epochs, batch_size, learning_rate, dataset_name, csv_file, log_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(batch_size, dataset_name)
    
    # Write the header for the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Gamma", "Decay Rate", "L0", "Accuracy", "Loss"])

    log_message(f'Start baseline optimizer', log_path)
    model = SimpleCNN().to(device)
    perform_baseline(model, device, num_epochs, train_loader, test_loader, criterion, log_path)
    log_message(f'End baseline optimizer', log_path)

    start_time = time.time()
    # Perform grid search
    for gamma in gamma_values:
        for decay_rate in decay_rate_values:
            model = SimpleCNN().to(device)
            log_message(f"Training with gamma={gamma}, decay_rate={decay_rate}", log_path)
            optimizer = LPAnnealingAdam(model.parameters(), alpha=decay_rate*10, decay_rate=decay_rate, start_lp=1.0, gamma=gamma)

            # Training loop
            for epoch in range(num_epochs):
                train_accuracy = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
                test_loss, test_accuracy = validate(model, device, test_loader, criterion, epoch)
                l0, num_params = get_model_complexity(model)
                message = (
                    f'Epoch {epoch + 1}/{num_epochs}, '
                    f'Gamma: {gamma}, Decay Rate: {decay_rate}, '
                    f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, '
                    f'Model Complexity: {l0}/{num_params}\n'
                    f'Elapsed Time: {format_time(time.time() - start_time)}'
                )
                log_message(message, log_path)
            
            # Write the results of the final epoch to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["LPAnnealingAdam", gamma, decay_rate, l0, test_accuracy, test_loss])
    log_message(f'Total Elapsed Time {format_time(time.time()-start_time)}', log_path)
    log_message(f"Grid search completed and results saved to {csv_file}", log_path)


# Define the grid search parameters
# gamma_values = [round(x * 0.05, 2) for x in range(20)] + [0.99, 0.999] # 22 values
# decay_rate_values = [round(1e-4 + x * 5e-5, 7) for x in range(21)] # 21 values
gamma_values = [0.95, 0.99, .995, 0.999] # 4 values # tune
decay_rate_values = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5] # 6 values # tune 
num_epochs = 10
batch_size = 64
learning_rate = 0.001 # tune
dataset_name = 'MNIST'
csv_file = 'results/grid_search_results.csv'
log_path = 'results/training_log.txt'

# Execute the grid search
perform_grid_search(gamma_values, decay_rate_values, num_epochs, batch_size, learning_rate, dataset_name, csv_file, log_path)