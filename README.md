#Automated plant stage labelling of herbarium samples in the family *Brassicaceae*


## Authors
[Matteo Jucker Riva](https://www.linkedin.com/in/matteojriva/), [Lindsey Viann Parkinson](https://www.linkedin.com/in/lindsey-viann/)

## Supervisors
[Badru Stanicki](https://www.linkedin.com/in/badrustanicki/), [Albin Plathottathil](https://www.linkedin.com/in/albin-plathottathil/), [Barry Sunderland](https://www.linkedin.com/in/barry-sunderland/)

## Purpose
To identify trends in the phenology of *Brassicaceae* by analyzing current digitized preserved specimens. Automatically label images of *Brassicaceae* specimens as flowering, fruiting, both or none. Potential to also use for new sample collections. 

## Data
The Herbaria Z+ZT and ZSS currently provide digital access to a total of 277,548 specimens which are all published under CC BY 4.0 licence and are accessible through the online portal:
[https://www.herbarien.uzh.ch/en/belegsuche.html](https://www.herbarien.uzh.ch/en/belegsuche.html )

Our study was limited to *Brassicaceae* samples collected from Valais, comprising approximately 6,000 images. Images are available from the herbarium online portal. However, some photos have been imcluded in the repo to allow the model to run through.

## Requirements
The requirements.txt file contains all the necessary python packages.
- TODO: hyperlink to requirements file

We ran the model using Google Colab. Some packages and dependencies may work better in Colab compared to other environments. 

## How to work with this repo

**IMPORTANT** : To make the model work please download:  
1. Mask_RCNN model files from [the AkTwelve's repo](https://github.com/akTwelve/Mask_RCNN) and add it to the src folder with the name "Mask_RCNN"  
2. Model weights from the following address: [model weights](https://drive.google.com/drive/folders/1HNs_EUyxMg8ThCRuseuJPSDYXR50GDNR?usp=sharing) and place it in the src folder with the name "model_weights"  
3. OPTIONAL annotated dataset from the following links: [train](https://drive.google.com/drive/folders/13Nph-NoTZwQFj-WOxwXtG61Wcf6fS9LR?usp=sharing), [test](https://drive.google.com/drive/folders/10-WqciDfjVAf5Qg6cJlHeigzvAWWDiQl?usp=sharing)



This repo is a customised version of the [Mask_RCNN port for tensorflow/keras built by Allen Kelly](https://github.com/akTwelve). In the **src** folder contains the main classes and functions needed to run this version of the model (**herbaria.py**),the pretrained weights of the best performing version of the model(**model weights**). Files for interacting and ussing the model are explained here below. 

### Command line interface

In the main project folder **preprocess_images.py** and **run_detection.py** allow CLI (command line interface) for easy access to important functions of the model. 
For example **run_detection** can be run on the terminal in the following way
1. cd path/to/project
2. `python3 run_detection.py -h` (to show an explanation for all the parameters)
3. `run_detection.py --input "path/to/images" --output "path/to/output/folder"` (example only, other parameters are available)
Preprocess_images

**1_resize_images.ipynb**  
Resizes images, scales annotations appropriately, and removes segments that have too few points

**2_Train_and_Inference.ipynb**  
We used Matterport's implementation of Mask R-CNN to train our dataset, then use the trained weights to run inference on new images

**3_Detect_Flowers_and_Fruits.ipynb**   
Demonstrates the process to use the trained model to detect fruits and flowers in herabrium sample pictures. 
For simplicity you can run the detect_brassicas script using a command line interface

**4_Evaluation_analysis.ipynb**   
Allows one to get predictions from a trained model and helps with understanding the results


#### Experimental Notebooks
This file has two note notebooks from our earlier experiments.

**InceptionResNetV2_classification.ipynb**  
Attempts to classify the images overall as fruiting flowering, both, or neither. It is missing a proper function to balance the classes which could improve results. At the time we stopped experimenting we were getting approximately 0.45 F1 score. 

**UNet_segmentation.ipynb**
Before trying MaskRCNN, versions of UNet were our best results. The UNet models I think have the potential to be as effective as MaskRCNN with further experimentation. WE had trouble converting the masks created by the model into useful classification metrics. 


### Preprocessing images
Image mask annotations were created with [Datatorch.io](https://www.datatorch.io) and provided here as a JSON file. 

### Running the model
We implemented a MaskRCNN model based off of the work by [akTwelve](https://github.com/akTwelve/tutorials/blob/master/mask_rcnn/MaskRCNN_TrainAndInference.ipynb) and [Matterport](https://github.com/matterport/Mask_RCNN). 

### Example Results
> ![name for image](./relative_path)  

- TODO add images to repo

Currently:
The model picks up 34% of reproductive structures. Of those it classifies 67% correctly as flowers or fruits. 

We believe significant improvements can be made to the Mask RCNN model with further parameter tuning. However, at the moment the model can still run through images and pull those with characteristics useful in phenological studies. Saving time for the initial data search. 
