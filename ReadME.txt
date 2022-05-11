This ReadME.txt file give setup and installation instrutions to run the code.

The code and ReadME.md file can also be found at https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets.

Code Structure:

The Zip file "" is structured as follows

|_
  |_ SRCNN/
     |_ run_srcnn.ipynb
     |_ Arun_srcnn.ipynb
     |_ model.py
     |_ dataset.py
     |_ Adataset.py
     |_losses.py
  |_ SRGAN/
     |_ 541-srgan.ipynb
     |_ 541-srresnet.ipynb
     |_ model.py
     |_ dataset.py
     |_ losses.py
  |_ downloaddiv2k.py
  |_ augument.ipynb
  |_ DataPreProcessing.ipynb
  |_ 541_psnr_ssim.ipynb.
  |_ results
  |_ README.md

SetUp Instruction : 

We trained differnt architectures for generating Super resolution images on google colab with Tesla K80 with 12gb ram. GPU is required for this project as we make use of mixed precision training, Gradient Scaling, GPU Cache clearing to accelerate the training process.


Installation Instruction | 

1) Download the dataset using downloaddiv2k.py.Thanks to https://github.com/Lornatang/SRGAN-PyTorch/blob/master/scripts/prepare_dataset.py for the script

2) Run DataPreProcessing.ipynb to split the data into train, test, validation set also to resize the image into (128, 128) for LR and (512, 512) for HR.

3) If you plan to run the model with data augumentation included. Run the DataPreprocessing.ipynb file 

4) For running differnt models. 
    SRCNN and DRRN:
         run the run_srcnn.ipynb inside SRCNN folder which is self contained for both training, validation and testing purposes.

    SRResNet:
         run the 541-srresnet.ipynb file from the SRGAN folder

    SRGAN:
         run the 541-srgan.ipynb file from the SRGAN folder.


5) few test results images are located inside the results folder

6) visualize and compare model outputs by running 541_psnr_ssim.ipynb file.

There will be few sections of code where the path for the dataset will be hard coded as we ran the model on google colab we had to use google drive to store images. Before running the Model make sure all the paths given inside the files have been updated to your local machines/ google drive location where the data will be stored.
References:

References for this project can be found in the below provided link
https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets.


