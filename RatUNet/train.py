from model import *
from dataprocess import *
import os
import nrrd
from utils import *
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import gc

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gc.collect()

torch.cuda.empty_cache()

def main():
    data_dir = "/Users/maggie-z/Desktop/Github/STD-Noise-Maps-of-CT-Images/scans"
    set_determinism(seed=4)

    # looping through for inspection
    count = 0
    dims = []
    sizes = []
    shapes = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.nrrd'):
            count += 1
            img, header = nrrd.read(os.path.join(data_dir,filename))
            dims.append(img.ndim)
            sizes.append(img.size)
            shapes.append(img.shape)
            
    dims_check = all(dim == dims[0] for dim in dims)
    size_check = all(size == sizes[0] for size in sizes)
    shape_check = all(shape == shapes[0] for shape in shapes)

    if dims_check and size_check and shape_check:
        print('Dimensions, shapes and sizes are uniform')
    else:
        print('Dimensions, shapes and sizes are NOT uniform')
        
    print('The total number of images in the dataset is {}'.format(count))

    custom_dataset = CustomData(root_dir = data_dir)
    train_dataset, val_dataset, test_dataset = custom_dataset[:568], custom_dataset[568:640], custom_dataset[640:]
    batch_size=1

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(len(train_loader), len(val_loader), len(test_loader))
    # Checking sizes for each DataLoader
    print_dataloader_sizes(train_loader, 'Train')
    print_dataloader_sizes(val_loader, 'Validation')
    print_dataloader_sizes(test_loader, 'Test')

    # Model and optimizer initialization
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # devise for Mac
    device = torch.device("mps")
    #model = RatUNet(BasicBlock, 64).to(device)
    model = RatUNet(BasicBlock, num_features=64, dropout_rate=0.0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    model_saving_path = './testresults/RatUNet_test_nodr.pth'

    # Train the model
    total_step = len(train_loader)
    #num_epochs = 300
    num_epochs = 2 # for testing environment
    best_val_loss = float('inf')  # Initialize best validation loss for model saving

    # training and validation

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        model.train()  # Ensure the model is in training mode

        for idx, images in enumerate(tqdm(train_loader)):
            img = images['ct_generated'].float().to(device)
            img = img.unsqueeze(0)
            std_map = images['std_map'].float().to(device)
            std_map = std_map.unsqueeze(0)
            #print(img.shape)
            #print(std_map.shape)
            optimizer.zero_grad()
            y_pred = model(img)
            #print(y_pred.shape)
            loss = average_relative_error(y_pred, std_map)
            #loss = criterion(y_pred, std_map)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for idx, image in enumerate(tqdm(val_loader)):
                img = image['ct_generated'].float().to(device)
                img = img.unsqueeze(0)
                std_map = image['std_map'].float().to(device)
                std_map = std_map.unsqueeze(0)

                y_pred = model(img)
                val_loss = average_relative_error(y_pred, std_map)
                #val_loss = criterion(y_pred, std_map)
                running_val_loss += val_loss.item() * img.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            # Save model if validation loss has improved
            if epoch_val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {epoch_val_loss:.6f}).  Saving model ...")
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), model_saving_path)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    # plot results
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses - without dropout')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.xticks(range(0, len(train_losses), 1))  # Specify tick locations every 5 epochs
    plt.legend()
    plt.show()

main()