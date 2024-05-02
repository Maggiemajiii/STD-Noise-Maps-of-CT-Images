# Standard Deviation Noise Maps of CT Scans using UNet and UNet-based Architectures

## Project Description
The goal of this project is to develop a deep learning model that can accurately estimate the standard deviation (STD) map of noise in CT scans. STD noise map is a per-pixel way to describe noise intensity and feature properties for finetuned maneuver of de-noising filters. We implemented a UNet model and a UNet-based model in order to predict STD noise maps from noisy CT scans.

## [Report](https://docs.google.com/document/d/1-uR0x-wku4VW0EU8N3Qm_44w2j1H_wEN_QM9PRKqIlY/edit?usp=sharing)

## Team Members
- Avantika Kothandaraman
- Caiwei Zhang
- Long Chen

## Folders in the Repository
**UNet** - Contains all the .py files needed to execute the UNet architecture

**RatUNet** - Contains all the .py files needed to execute the RatUNet architecture

**notebooks** - Contains Jupyter Notebook for both models



## How to Run the Project
To reproduce the results:

### **Prepare the Environment**: 

Ensure all required packages are installed.

To run both UNet and RatUNet, you need to install the necessary Python packages. You can install these packages using pip in the terminal/command prompt or in an interactive environment like Jupyter Notebooks:

```bash
pip install numpy nibabel matplotlib torch torchvision pynrrd scipy scikit-learn SimpleITK patchify tqdm
```

We would personally suggest using an interactive environment. 

#### Utilities:

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

## Models:

### UNet 

To train the [U-Net model](https://arxiv.org/abs/1505.04597) , run 

```
python3 ./UNet/train_and_test.py
```

This script will train the U-Net model using the training data loaded via `DataLoader`. This model is set default to run on CUDA device, by changing that label according to your platform can you run this model on your device.

[Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with `learning_rate = 0.0001` is set as the default training optimizer for UNet.

#### RatUNet

To train the [RatUNet model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9138094/),  run 

```
python3 ./RatUNet/train_and_test.py
```

This script will train the RatUNet model with `DataLoader`prepares the data. 

The RatUNet model, a variant of the UNet architecture enhanced with  residual blocks and attention mechanism. The model is set default to run on an MPS device, by changing that label according to your platform can you run this model on your device.

[Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with `learning_rate = 0.00001` is set as default training optimizer for UNet.

## Training and Validation

Use script `train.py` in both directories to perform The model undergoes training over a specified number of epochs, where in each epoch:

- The model processes batches of CT images, computes a loss (here, the average relative error between the predictions and the actual STD maps), and updates the model weights accordingly.
- The script also evaluates the model on the validation dataset after each training epoch to monitor its performance on unseen data. If the validation loss improves, the model state is saved, facilitating model performance tracking and recovery.
- After training, the script plots training and validation losses across  epochs, providing visual feedback on the learning process and model convergence.

Remember to change the data directory in `train.py` or `train_and_test.py` to where the data is located: 
```
data_dir = 'path/to/scans'
```

### **Prediction Visualization**

Use script `train_and_test.py` to train, visualize, and compare the predicted result and ground truth.

- The file undergoes the same training process as `train.py`
- After training, the model would be evaluated on a test dataset where final predictions are compared against true data. The results are visualized to demonstrate the model's capability to predict noise maps from medical images.


