# alpha static-lp

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

BATCH_SIZE = 64
NUM_EPOCHS = 15
WEIGHTS = {
    "val_loss"=0.3,
    "val_accuracy"=0.6,
    "sparsity"=0.1
}

log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "StaticLPAnnealingAdam3.log")),
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

def objective(trial):
    # Hyperparameter search space
    alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)

    logger.info(f"Starting trial {trial.number}: alpha={alpha}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model instantiation
    model = SimpleCNN().to(device)  # Ensure SimpleCNN accepts dropout_rate

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = LPAnnealingAdam(model.parameters(), alpha=alpha, start_lp=1.0, end_lp=1.0)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss  # Modify if LPAnnealingAdam adds regularization

            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}")

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
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - "
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Compute sparsity
        num_params = sum(p.numel() for p in model.parameters())
        num_nonzero = sum(p.nonzero().size(0) for p in model.parameters())
        sparsity = 1.0 - (num_nonzero / num_params)
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS} - Sparsity: {sparsity:.4f}")
    return val_loss, val_accuracy, sparsity

# def run_optimization(n_trials=50):
#     logger.info(f"Starting hyperparameter optimization with {n_trials} trials.")
#     pruner = optuna.pruners.MedianPruner()
#     study = optuna.create_study(direction='minimize', pruner=pruner)
#     study.optimize(objective, n_trials=n_trials, timeout=None)

#     logger.info("Optimization finished.")
#     logger.info(f"Number of finished trials: {len(study.trials)}")
#     logger.info("Best trial:")
#     trial = study.best_trial

#     logger.info(f"  Value: {trial.value}")
#     logger.info("  Params: ")
#     for key, value in trial.params.items():
#         logger.info(f"    {key}: {value}")

#     return study

# # Run the optimization
# study = run_optimization(n_trials=2)

# def run_multi_objective_optimization(n_trials=50):
#     logger.info(f"Starting multi-objective hyperparameter optimization with {n_trials} trials.")
#     study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'])  # Minimize loss, maximize sparsity
#     study.optimize(objective, n_trials=n_trials)

#     logger.info("Multi-objective optimization finished.")
#     logger.info(f"Number of finished trials: {len(study.trials)}")
#     logger.info("Best trials:")
#     for trial in study.best_trials:
#         logger.info(f"  Trial {trial.number} - Combined Objective: {trial.values}")
#         logger.info("  Params: ")
#         for key, value in trial.params.items():
#             logger.info(f"    {key}: {value}")

#     return study

# Run the multi-objective optimization
study = run_multi_objective_optimization(n_trials=50)

# Analyze results
# logger.info("Best hyperparameters: ")
# for key, value in study.best_params.items():
#     logger.info(f"  {key}: {value}")

try:
    fig1 = vis.plot_optimization_history(study)
    fig1.savefig(os.path.join(log_dir, "static_optimization_history2.png"))

    fig2 = vis.plot_param_importances(study)
    fig2.savefig(os.path.join(log_dir, "static_param_importances2.png"))

    fig3 = vis.plot_slice(study)
    fig3.savefig(os.path.join(log_dir, "static_slice_plot2.png"))

    logger.info("Visualization plots saved successfully.")
except Exception as e:
    logger.error(f"Visualization failed: {e}")
