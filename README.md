# Standard Deviation Noise Maps of CT Scans using UNet and UNet-based Architectures

## Project Description
The goal of this project is to develop a deep learning model that can accurately estimate the standard deviation (STD) map of noise in CT scans. STD noise map is a per-pixel way to describe noise intensity and feature properties. We implemented a UNet model and a UNet-based model in order to predict STD noise map from noisy CT scans.

## Team Members
- Avantika Kothandaraman
- Caiwei Zhang
- Long Chen

## Folders in the Repository
1. **UNet** - Contains all the .py files needed to execute the UNet architecture
2. **RatUNet** - Contains all the .py files needed to execute the RatUNet architecture

## Installation
To run both UNet and RatUNet, you need to install the necessary Python packages. You can install these packages using pip in the terminal/command prompt or in an interactive environment like Jupyter Notebooks:

```bash
pip install numpy nibabel matplotlib torch torchvision pynrrd scipy scikit-learn SimpleITK patchify tqdm
```
We personally suggest using an interactive environment. 

## How to Run the Project
To reproduce the results:

#### **Prepare the Environment**: 

Ensure all required packages are installed.

#### **Imports**: 

Run UNet/imports.py first before beginning any code execution (applies to both UNet and RatUNet)

### UNet:

#### **Data Loading**: 

The `custom_dataset.py` script is designed to handle data loading. Initialize this script with the directory containing your CT images. Your data directory should contain images in the following format:

- `L056_signal.nrrd`
- `L056_noise.nrrd`
- `L056_std.nrrd`

Ensure that the images follow this naming convention. If your images are stored in a different format, you will need to modify `custom_dataset.py` to accommodate your specific file naming convention.

#### **Model Training and Testing**: 

To train the U-Net model, first include `unet.py` from the same directory as the training network, then navigate to the directory containing `train.py` and run. This script will train the U-Net model using the training data loaded via `custom_dataset.py`. Ensure that you have adjusted the paths and parameters in `unet_model_loss_training.py` according to your setup.

After training, you can test the U-Net model's performance on unseen data by running: `unet_average_percentage_error.py`

This script will load the trained U-Net model and test it against your  test dataset. It will output the performance metrics defined in your  testing script, such as Average Relative Error or any other metric you  have defined.



#### Visualization of Results:

For a visual assessment of the model's performance, you can use `unet_testing.py` where noisy images would be evaluated and prediction would be made. 

After that, use script`unet_plots_patch_reconstruction.py` to reconstruct predicted patches back to full scale 512x512 slice of predicted image.

### RatUNet:

To reproduce the results:

#### Enable Utilities:

The `utils.py` includes several crucial functions essential for processing and evaluating deep learning models:

`convert_to_2d()`

This function compresses three-dimensional image data into two-dimensional images, making it ideal for extracting specific views from 3D medical scans, such as producing cross-sectional images from MRI or CT scans. This capability is a common requirement in medical image analysis.

`average_relative_error()`

This function calculates the average relative error between model outputs and targets, commonly used as a loss function during training. It assists in optimizing model parameters to reduce prediction errors, critical for assessing model performance and adjusting model architecture.

`print_dataloader_sizes()`

This function prints important information about data loading, such as the total number of samples, number of batches, and batch sizes. This functionality is vital for ensuring data is correctly loaded and batch-processed, aiding in debugging and optimizing the data loading process.

The proper implementation and application of these tools and functions are indispensable for ensuring the effectiveness and efficiency of model training. They serve as a bridge in the project codebase, ensuring precision in data handling and accuracy in model evaluation.

#### **Data Loading**: 

The `dataprocess.py` script is designed to handle data loading. Initialize this script with the directory containing your CT images. Your data directory should contain images in the following format:

- `L056_signal.nrrd`
- `L056_noise.nrrd`
- `L056_std.nrrd`

Ensure that the images follow this naming convention. If your images are stored in a different format, you will need to modify `dataprocess.py` to accommodate your specific file naming convention.

`class CustomData` 

An essential part of our data handling framework, this subclass of `torch.utils.data.Dataset` automates the loading, processing, and patching of image data from a specified directory. It handles:

- Reading NRRD files for signal, noise, and standard deviation maps.
- Combining signal and noise data to create synthetic CT images influenced by variable noise levels.
- Extracting small patches from these synthetic images, which are suitable for training deep learning models.

- `data_info`: Provides detailed insights into the data items, including their shapes and data types, which is vital for debugging and ensuring data integrity.
- `plot_ct`: A visualization utility that plots patches of CT images alongside their corresponding noise maps, allowing for immediate visual assessment of the preprocessing steps.

#### **Model Training and Testing**: 

To train the RatUNet model, navigate to the directory containing `train.py` and run. This script will train the RatUNet model with `DataLoader`prepares the data.

- It reads medical images in NRRD format from a specified directory, checking their dimensions, sizes, and shapes to ensure uniformity, which is crucial for consistent processing.
- A custom dataset object (`CustomData`) is created, splitting the dataset into training, validation, and test subsets which are then loaded using PyTorch’s `DataLoader` for efficient batch processing during model training.

The RatUNet model, a variant of the UNet architecture enhanced with  features like BasicBlocks and dropout for robust feature learning, is  initialized.  The model is set default to run on an MPS device, by changing that label according to your platform can you run this model on your device.

This script will load the trained U-Net model and test it against your  test dataset. It will output the performance metrics defined in your  testing script, such as Average Relative Error or any other metric you  have defined.

###### Training and Validation

The model undergoes training over a specified number of epochs, where in each epoch:

- The model processes batches of CT images, computes a loss (here, the average relative error between the predictions and the actual STD maps), and updates the model weights accordingly.
- The script also evaluates the model on the validation dataset after each training epoch to monitor its performance on unseen data. If the validation loss improves, the model state is saved, facilitating model performance tracking and recovery.

###### **Results Visualization**

After training, the script plots training and validation losses across  epochs, providing visual feedback on the learning process and model convergence.

