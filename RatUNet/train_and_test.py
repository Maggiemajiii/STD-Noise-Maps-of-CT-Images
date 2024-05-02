from ratunet import *
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
    data_dir = "/projectnb/ec500kb/projects/Project6/scans"
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
    model_saving_path = './testresults/RatUNet_test_nodr.pth'

    # Train the model
    total_step = len(train_loader)
    #num_epochs = 300
    num_epochs = 2 # for testing environment
    best_val_loss = float('inf') 

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

            optimizer.zero_grad()
            y_pred = model(img)

            loss = average_relative_error(y_pred, std_map)
            #loss = criterion(y_pred, std_map)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()  
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

    model.eval()
    test_loss = 0.0
    test_image = []
    ground_truth = []
    predicted_patches = []

    model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            # moving data to device
            images = batch['ct_generated'].float().to(device)
            std_maps = batch['std_map'].float().to(device)
            
            images = images.unsqueeze(0)
            std_maps = std_maps.unsqueeze(0)

            y_pred = model(images)

            # calculating loss
            loss = average_relative_error(y_pred, std_maps)
            #loss = criterion(y_pred, std_maps)
            test_loss += loss.item() * images.size(0)
            y_pred = y_pred.squeeze(0)

            # appending data for visualization
            test_image.append(images.cpu().numpy())
            ground_truth.append(std_maps.cpu().numpy())
            predicted_patches.append(y_pred.cpu().numpy())

    # calculating average test loss
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # visualizing prediction

    test_image_array = np.array(test_image)
    ground_truth_array = np.array(ground_truth)
    predicted_patches_array = np.array(predicted_patches)

    test_image_array = test_image_array.reshape(8,8,64,64)
    ground_truth_array = ground_truth_array.reshape(8,8,64,64)
    predicted_patches_array = predicted_patches_array.reshape(8,8,64,64)

    test_image_final = np.concatenate([np.concatenate(row, axis=1) for row in test_image_array], axis=0)
    ground_truth_final = np.concatenate([np.concatenate(row, axis=1) for row in ground_truth_array], axis=0)
    predicted_patches_final = np.concatenate([np.concatenate(row, axis=1) for row in predicted_patches_array], axis=0)
    # Save the predicted_patches_final array to a file
    #np.save('./results/seed104_nodr_predict.npy', predicted_patches_final)
    # Plot the test image
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(np.rot90(test_image_final), cmap='gray')
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(np.rot90(ground_truth_final), cmap='hot')
    plt.title('Ground Truth Noise Map')
    plt.axis('off')


    plt.subplot(133)
    plt.imshow(np.rot90(predicted_patches_final), cmap='hot')
    plt.title('Predicted Noise Map')
    plt.axis('off')
    plt.show()
    """
    # Load the predicted_patches_final array from a file to get average prediction
    p4 = np.load('./results/seed4_predict.npy')
    p54 = np.load('./results/seed54_predict.npy')
    p104 = np.load('./results/seed104_predict.npy')
    average_prediction = (p4 + p54 + p104) / 3

    # Plot the average prediction
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(np.rot90(test_image_final), cmap='gray')
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(np.rot90(ground_truth_final), cmap='hot')
    plt.title('Ground Truth Noise Map')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(np.rot90(average_prediction), cmap='hot')
    plt.title('Predicted Noise Map avg of 3')
    plt.axis('off')
    plt.show()

    #show average prediction evaluation
    percent_error = (np.array(ground_truth_final) - np.array(average_prediction)) / np.array(ground_truth_final)
    average_percent_error = abs(np.mean(np.abs(percent_error)))
    print(average_percent_error)

    # percentage error visualization
    epsilon = 1e-8
    ground_truth_final_safe = ground_truth_final + (ground_truth_final == 0) * epsilon

    # Calculate the percentage error
    percent_error = (ground_truth_final - average_prediction) / ground_truth_final_safe
    average_percent_error = np.mean(np.abs(percent_error)) * 100  # Convert to percentage

    print(f"Average Percentage Error: {average_percent_error:.2f}%")

    # Plotting the percentage error as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(np.rot90(percent_error), cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Percentage Error (%)')
    plt.title('Percentage Error Heatmap')
    plt.axis('off')  # Hide the axes
    plt.show()
    """

main()
