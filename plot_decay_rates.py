import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a .csv file
date = '11_24_2024'
file_path = f'logs/optimization_results_{date}.csv'
out_folder = 'plots/'
data = pd.read_csv(file_path)

# # Plot decay_rates vs alpha
# plt.figure(figsize=(15, 6))
# plt.plot(data['decay_rate'], data['alpha'], label='Decay Rate vs Alpha')
# plt.xlabel('Decay Rate')
# plt.ylabel('Alpha')
# plt.title('Decay Rate vs Alpha')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'{out_folder}decay_rate_vs_alpha.png')

# # Plot decay_rates vs gamma
# plt.figure(figsize=(15, 6))
# plt.plot(data['decay_rate'], data['gamma'], label='Decay Rate vs Gamma', color='red')
# plt.xlabel('Decay Rate')
# plt.ylabel('Gamma')
# plt.title('Decay Rate vs Gamma')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'{out_folder}decay_rate_vs_gamma.png')

# # Plot decay_rates vs sparsity
# plt.figure(figsize=(15, 6))
# plt.plot(data['decay_rate'], data['sparsity'], label='Decay Rate vs Sparsity', color='green')
# plt.xlabel('Decay Rate')
# plt.ylabel('Sparsity')
# plt.title('Decay Rate vs Sparsity')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'{out_folder}decay_rate_vs_sparsity.png')


# # Plot decay_rates vs accuracy
# plt.figure(figsize=(15, 6))
# plt.plot(data['decay_rate'], data['val_accuracy'], label='Decay Rate vs Accuracy', color='blue')
# plt.xlabel('Decay Rate')
# plt.ylabel('Accuracy')
# plt.title('Decay Rate vs Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'{out_folder}decay_rate_vs_accuracy.png')


# LOG X AXIS

plt.figure(figsize=(15, 6))
plt.plot(data['decay_rate'], data['alpha'], label='Decay Rate vs Alpha', color='purple')
plt.xscale('log')
plt.xlabel('Decay Rate (Log Scale)')
plt.ylabel('Alpha')
plt.title('Decay Rate vs Alpha (Logarithmic Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{out_folder}log_decay_rate_vs_alpha_{date}.png')

# Plot decay_rates vs gamma
plt.figure(figsize=(15, 6))
plt.plot(data['decay_rate'], data['gamma'], label='Decay Rate vs Gamma', color='red')
plt.xscale('log')
plt.xlabel('Decay Rate (Log Scale)')
plt.ylabel('Gamma')
plt.title('Decay Rate vs Gamma (Logarithmic Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{out_folder}log_decay_rate_vs_gamma_{date}.png')

# Plot decay_rates vs sparsity
plt.figure(figsize=(15, 6))
plt.plot(data['decay_rate'], data['sparsity'], label='Sparsity', color='green')
plt.xscale('log')
plt.xlabel('Decay Rate (Log Scale)')
plt.ylabel('Sparsity')
plt.title('Decay Rate vs Sparsity (Logarithmic Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{out_folder}log_decay_rate_vs_sparsity_{date}.png')


# Plot decay_rates vs accuracy
plt.figure(figsize=(15, 6))
plt.plot(data['decay_rate'], data['val_accuracy'], label='Decay Rate vs Accuracy', color='blue')
plt.xscale('log')
plt.xlabel('Decay Rate (Log Scale)')
plt.ylabel('Accuracy')
plt.title('Decay Rate vs Accuracy (Logarithmic Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{out_folder}log_decay_rate_vs_accuracy_{date}.png')

plt.figure(figsize=(15, 6))
plt.plot(data['decay_rate'], data['val_accuracy'], label='Accuracy', color='blue')
plt.plot(data['decay_rate'], data['sparsity'], label='Sparsity', color='green')
plt.plot(data['decay_rate'], data['gamma'], label='Gamma', color='red')
plt.plot(data['decay_rate'], data['alpha'], label='Alpha', color='purple')
plt.xscale('log')
plt.xlabel('Decay Rate (Log Scale)')
plt.ylabel('All!')
plt.title('Decay Rate vs All metrics (Logarithmic Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{out_folder}log_decay_rate_vs_all_{date}.png')