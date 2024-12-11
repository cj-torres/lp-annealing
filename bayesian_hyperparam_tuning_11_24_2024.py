import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna
import optuna.visualization as vis
from lp_optim import LPAnnealingAdam
from SimpleCNN import SimpleCNN
import os
import numpy as np
import pandas as pd

BATCH_SIZE = 64
NUM_EPOCHS = 30
DECAY_RATES = np.logspace(-6, -3, num=60)

log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "LPAnnealingAdamDecayRate_11_24_2024.log")),
        logging.StreamHandler()
    ]
)

# Create a logger object
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

logger.info("Loading MNIST dataset...")
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logger.info(f"Dataset split into {train_size} training and {val_size} validation samples.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# DataFrame to store intermediate results
results_df = pd.DataFrame(columns=['decay_rate', 'alpha', 'gamma', 'val_loss', 'val_accuracy', 'sparsity'])

def objective(trial, decay_rate):
    # Hyperparameter search space
    alpha = trial.suggest_float('alpha', 1e-4, 0.017, log=True)
    gamma = trial.suggest_float('gamma', 1e-4, 1e-2, log=True)

    logger.info(f"Decay Rate: {decay_rate} | Trial {trial.number}: alpha={alpha}, gamma={gamma}")

    # Model instantiation
    model = SimpleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = LPAnnealingAdam(model.parameters(), alpha=alpha, start_lp=1.0, gamma=gamma, decay_rate=decay_rate)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        logger.info(f"Decay Rate: {decay_rate} | Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        logger.info(f"Decay Rate: {decay_rate} | Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Compute sparsity
        num_params = sum(p.numel() for p in model.parameters())
        num_nonzero = sum(p.nonzero().size(0) for p in model.parameters())
        sparsity = 1.0 - (num_nonzero / num_params)
        logger.info(f"Decay Rate: {decay_rate} | Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - Sparsity: {sparsity:.4f}")
    return val_loss

def run_optimization(decay_rates, n_trials=50):
    for decay_rate in decay_rates:
        logger.info(f"Starting optimization for decay_rate={decay_rate}")

        # Define a partial function to include decay_rate in the objective
        func = lambda trial: objective(trial, decay_rate)

        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(func, n_trials=n_trials, timeout=None)

        logger.info(f"Optimization finished for decay_rate={decay_rate}")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Validation Loss: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        # Extract metrics from the best trial
        best_alpha = trial.params['alpha']
        best_gamma = trial.params['gamma']

        # To retrieve accuracy and sparsity, we need to run the training again with the best hyperparameters
        logger.info(f"Evaluating best trial for decay_rate={decay_rate}")

        # Instantiate the model with best hyperparameters
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = LPAnnealingAdam(model.parameters(), alpha=best_alpha, start_lp=1.0, gamma=best_gamma, decay_rate=decay_rate)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss = loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

            epoch_loss /= len(train_loader.dataset)

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_accuracy = correct / total

            # Compute sparsity
            num_params = sum(p.numel() for p in model.parameters())
            num_nonzero = sum(p.nonzero().size(0) for p in model.parameters())
            sparsity = 1.0 - (num_nonzero / num_params)

        # Record the results
        results_df.loc[len(results_df)] = [decay_rate, best_alpha, best_gamma, val_loss, val_accuracy, sparsity]
        logger.info(f"Recorded results for decay_rate={decay_rate}")

    return results_df

# Run the optimization
results = run_optimization(DECAY_RATES, n_trials=50)

# Save the results to a CSV file
results.to_csv(os.path.join(log_dir, "optimization_results_11_24_2024.csv"), index=False)
logger.info("Intermediate results saved to optimization_results_11_24_2024.csv")

# Plotting
try:
    import matplotlib.pyplot as plt

    # Plot decay rates vs alpha
    plt.figure(figsize=(8,6))
    plt.loglog(results['decay_rate'], results['alpha'], marker='o')
    plt.xlabel('Decay Rate')
    plt.ylabel('Alpha')
    plt.title('Decay Rate vs Alpha')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(log_dir, "decay_rate_vs_alpha_11_24_2024.png"))
    plt.close()
    logger.info("Plot saved: decay_rate_vs_alpha_11_24_2024.png")

    # Plot decay rates vs gamma
    plt.figure(figsize=(8,6))
    plt.loglog(results['decay_rate'], results['gamma'], marker='o', color='orange')
    plt.xlabel('Decay Rate')
    plt.ylabel('Gamma')
    plt.title('Decay Rate vs Gamma')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(log_dir, "decay_rate_vs_gamma_11_24_2024.png"))
    plt.close()
    logger.info("Plot saved: decay_rate_vs_gamma_11_24_2024.png")

except Exception as e:
    logger.error(f"Plotting failed: {e}")

logger.info("Optimization and plotting completed successfully.")