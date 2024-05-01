import numpy as np
import torch

def convert_to_2d(img_volume, axis=1):
    return np.max(img_volume, axis=axis)

# LOSS Function - Average Relative Error
def average_relative_error(output, target, epsilon=1e-8):
    absolute_error = torch.abs(output - target)
    # Adding epsilon to the denominator to avoid division by zero
    relative_error = absolute_error / (torch.abs(target) + epsilon)
    return torch.mean(relative_error)

def print_dataloader_sizes(loader, name):
    # Total number of samples
    total_samples = len(loader.dataset)
    
    # Number of batches
    num_batches = len(loader)
    
    # Assuming all batches have the same size except possibly the last one
    batch_size = loader.batch_size
    
    print(f'{name} DataLoader has:')
    print(f'  Total samples: {total_samples}')
    print(f'  Number of batches: {num_batches}')
    print(f'  Batch size: {batch_size}')
    if total_samples % batch_size != 0:
        print(f'  Last batch size: {total_samples % batch_size}')
    else:
        print(f'  All batches have the same size')
