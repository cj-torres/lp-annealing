import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna
import optuna.visualization as vis
from getSimpleCNNWithL0 import getSimpleCNNWithL0

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/BayesianHyperparameterTuningTraining.log"),
        logging.StreamHandler()
    ]
)

# Create a logger object
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

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
    lam = trial.suggest_float('lam', 1e-4, 1e-1, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 20)

    logger.info(f"Starting trial {trial.number}: lam={lam}, lr={lr}, batch_size={batch_size}, "
                f"dropout_rate={dropout_rate}, weight_decay={weight_decay}, num_epochs={num_epochs}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model instantiation
    model = getSimpleCNNWithL0(
        lam=lam,
        weight_decay=weight_decay,
        droprate_init=dropout_rate  # Assuming droprate_init corresponds to dropout_rate
    ).to(device)

    # Update dropout rate in the model
    model.module.dropout = nn.Dropout(dropout_rate)
    logger.debug("Dropout rate updated in the model.")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l0_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l0_loss = model.regularization()
            total_loss = loss + l0_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_l0_loss += l0_loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        epoch_l0_loss /= len(train_loader.dataset)
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {epoch_loss:.4f}, L0 Loss: {epoch_l0_loss:.4f}")

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
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} - "
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Report intermediate objective value
        trial.report(val_loss, epoch)

        # Prune trials that are not promising
        if trial.should_prune():
            logger.warning(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number} completed with Validation Loss: {val_loss:.4f}")
    return val_loss  # Or return -val_accuracy if maximizing accuracy

def run_optimization(n_trials=50):
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials.")
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=None)

    logger.info("Optimization finished.")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    return study

# Run the optimization
study = run_optimization(n_trials=2)

# Analyze results
logger.info("Best hyperparameters: ")
for key, value in study.best_params.items():
    logger.info(f"  {key}: {value}")

# Visualization (optional, may require a Jupyter environment)
try:
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()
except Exception as e:
    logger.error(f"Visualization failed: {e}")
