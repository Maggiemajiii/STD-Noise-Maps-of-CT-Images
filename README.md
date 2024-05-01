# Standard Deviation Noise Maps of CT Scans using UNet and UNet-based Architectures

#### Project Description
The goal of this project is to develop a deep learning model that can accurately estimate the standard deviation (STD) map of noise in CT scans. We implemented a UNet model and a UNet-based model in order to reach our goal

#### Team Members
- Avantika Kothandaraman
- Caiwei Zhang
- Long Chen

#### Folders in the Repository
1. **UNet** - Contains all the .py files needed to execute the UNet architecture
2. **RatUNet** - Contains all the .py files needed to execute the RatUNet architecture

#### Installation
To run both UNet and RatUNet, you need to install the necessary Python packages. You can install these packages using pip in the terminal/command prompt or in an interactive environment like Jupyter Notebooks:

```bash
pip install numpy matplotlib torch torchvision pynrrd SimpleITK patchify
```
We personally suggest using an interactive environment. 

#### How to Run the Project
To reproduce the results:
1. **Prepare the Environment**: Ensure all required packages are installed.
2. **Imports**: Run UNet/imports.py first before beginning any code execution (applies to both UNet and RatUNet)
3. **Data Loading**: custom_dataset.py is designed to be initialized with the directory of interest containing all the images necessary. The images must be stored in this example format:
   'L056_signal.nrrd', 'L056_noise.nrrd', 'L056_std.nrrd'. If the images are not in this expected naming convention or format, the custom_dataset.py must be modified to load and process 
   the files as per the desired naming convention. 
4. **Model Training and Testing**:

