# Development log of herbaria model


## Work plan (updated 26.10)
### Preprocessing:
#### Getting the outline of plants
	Libraries to try. https://opensource.com/article/19/3/python-image-manipulation-tools

#### ?? Get the structure of specimen (in order to localize better flowers and fruits)
- Interpolate pixels under the tape  
- Translate outline into simple vector/graph structures OR

#### Localise “hot” areas for flower and fruits
e.g following https://github.com/matterport/Mask_RCNN & https://www.frontiersin.org/articles/10.3389/fpls.2020.01129/full

### Labelling training data:    
Possibilities:
- Manual labelling of 200-300 pictures   
- Active learning ??   
- One shot learning?   
- Use date and location as fuzzy classifier?   
Define Class balance / species balance    
E.g. Number of flowers, Number of stems with fruits    

### Classification:
 From project description
> Obviously, you should score a herbarium sheet (i) as flowering, when you can ‘clearly’ distinguish flowers, (ii) as fruiting, when you can ‘clearly’ distinguish fruits (siliques or silicles, depending on the species), (iii) as both flowering and fruiting, when you can ‘clearly’ distinguish flowers and fruits on a given voucher (see ZT-00142092 or ZT-00142096 as an example), and (iv) at the vegetative stage, when you cannot distinguish either flowers or fruits on a given voucher, but solely leaves and stems. Caution, there is a time-point, right after pollination, when flowers have just wilted, but fruits have already started to develop; under such circumstances, petals may still be apparent, but fruits may already protrude out of the flowers. I suggest to score such specimens as ‘ambiguous’, because neither flowers, nor fruits can be ‘clearly’ distinguished (see specimen ZT-00081542 as an example).

#### Multilabel classification with 3 categories:   
- Flowering: flowers present on the top on the stems   
- Fruit: fruit pods / Elongated or round pods along the main stem   
- Both fruits and flower: dixit  
- None: vegetative state (possibly under-represented category),  
- Ambiguous: in transition between flowers and fruits   

### Interpretation:  
- Map with location and phenology info? 



## Log

=======
## Daily Outline

### October 26
- Started reviewing provided data and goals of the project
- Created questions for meeting with Barry tomorrow
- Background research on other herbarium classification tasks
- discussed possible shortcomings of images and technology
- Completed first attempt to define a bounding box using morphological transformations and masks

TODO 
- decide how we want to label the photos
- make plan of action which models to try first

=======
### October 27
Conclusions from yesterday:
- Bounding box detection should be possible using openCV workflow (no ML), to be tested more
- Pytorch project for labelling the images (exact labelling proceure is still to be defined)

Meeting with Barry at 10am.
decide daily workflow and project goals after

Post meeting TODO:
- Google cloud free service
- Run trial models with public labeled dataset from paper
- Label photos 
- Export labels in COCO format
- Add overall classification to image metadata
- create bounding box workflow and ensure it works with full res images

=============
### October 28
TODO
- image segmentation of 20 - 30 photos
- Should create over 1000 instances of reproductive structures
- Solve problem in accessing data from google cloud. At the moment authorisation seem not to be working

DONE
- Completed preprocessing function to get the outline of plants (to be tested again with more images)
- Added a notebook with the tries I made to access the photos

=============
### October 29

TODO
- Build model and train on public dataset
- Understand workflow for training and fitting (COLAB vs Jupyter)
- Save images on Google drive???
- Understand exact model architecture for instance segmentation: 
        https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1 
- Understand how to import and visualise masks stored as json file
        https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/cocostuff/pngToCocoResultDemo.py
        https://github.com/cocodataset/cocoapi/issues/111

DONE
- Found external database with images and masks to use for instance segmentation training
- https://competitions.codalab.org/competitions/18405#learn_the_details
- https://www.plant-phenotyping.org/datasets-overview
- https://projects.asl.ethz.ch/datasets/doku.php?id=2018plantstressphenotyping
- https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-hfr.336.1
- Completed segmentation of eth images (32 with 1445 annotations)
- contacted DataTorch CEO to help with exports
- Created function to download images form Google Cloud
- Began creating script to classify images overall (without segments)

Trouble with upsampling/balancing function
================

### October 30
TODO
- finish and test classification script
- need to link names from classification csv to names of photos in Drive
- make sure json file of annotations is working


DONE
- Understood how to use training data form knb dataset
- Downloaded json file with all annotations from DataTorch
- Loaded centerNet hourglass in google colab and obtained *wrong* predictions
- implement OBJECT RECOGNITION model CENTERNET HOURGLASS in COLAB notebook 011
- Ran basic classification model on ResNet50 architecture
    * used images without annotations
    * terrible results due to resizing too small
    * tried opening initial layers to allow 1024x600 or 2048x1024
        + include_top = False
        + does Tensorflow have layer like AdaptiveAvgPool2D?
-  sent message to DJ for advice aboue small validation datasets

=======================

### October 31
- DJ agreed to help but haven't heard back specifics. It is the weekend after all :)

========================

### November 2
WEEKLY GOALS
- working classification model
- working segmentation model
- fine tune models
- fine tune training data generation
- test model(s) with annotated UZH herbarium photos

TODO
- Matteo: fine tuning with KNB dataset
- Lindsey: get high res images into ResNet?
    * try model structures other than ResNet?
    * Matteo using CenterNet Hourglass, would that work on classification?
- Organize questions and updates for Tuesday morning meeting

DONE
- From DJ:
- 1. About sample size:
    * a. If you have time it would be better to hand-classify a bit more because supervised models always beat unsupervised models with more complex models no matter how sophisticated it is
    * b. Try to use more complex approaches like self-supervised learning to first train on the unsupervised labeled data, then fine-tune on the smaller labeled dataset and try predictions (methods include SimCLR very recently released from google)
    *c. You can try training on less data using methods like few shot learning e.g siamese networks or even newer models like Big Transfer ( it is a bit computationally expensive but here is an example where I had trained on total 25-30 images as an experiment: https://colab.research.google.com/drive/1CXvQZ2gfhUr6zrv4DJ_hqN-4xOmGdIue ). Let me know in case you cant view the notebook.
- 2. About pgoto resolution: You can use any resolution from what I remember just set the base network to include_top=False and it should be useful. Average pooling you can add in anyway after you get the last conv\pool layer from resnet (refer to the tutorial notebook it should have the example with include top as False)

- inputing larger images still problematic
- attempted CenterNet. Had issues with tensor shape
    * are we mixing up image shape and tensor shape at some points in the code?
    * CenterNet throws errors but do other models loaded in the same fashion have similar errors

========================

### November 2

DONE:
- Built model with centerNet and functional API.
- Checked with Alessia Gubbisberg (ETHZ) about phenology indicators and final model outputs


TODO:
- Transform the data in TFRecordfile, the format to feed to the model. See : https://blog.roboflow.com/create-tfrecord/  
- merge code from butterfly and bird classification programs 

========================
### November 3
- Tensorflow is problematic
- Met with Dr. Alessia Guggisberg
    * main goal: overall classification
    * secondary goal: reproduction ratios
        + seeds:flowers
        + reproductive pixels:biomass pixels
        + plant pixels : image size (for size of plant)
    * tertiary: maps and any extras we can think of
========================
### November 4

DONE:
- Found guide on how to implement finw tuning of object detction models with pyTorch
- converted bird classification code to plant classification code
- scheduled meeting with Barry (and Badru?) 10:30am tomorrow
- Met with Nitin
    * Can we cut images randomly? and supply as individual images?
    * Mix few-shot and semi-supervised learning
    * review cancer MRI notebooks on Github
    * train ResNet, VGG, EfficientNet, etc. in parallel
    * Do the models make mistakes on the same or different images?
        + if same, only use one
        + if different, extract features and combine
    * use stepwise approach to problem solving - start simple and work up
- got image augmentation to work (more or less)

TODO: 
- Run mask RCNN with coco dataset to evaluate preprocessin
- Implement object detection model using pytorch (notebook 13)
- create properly balanced image augmentations using imgaug package
    * can currently output images but is not efficient and not balanced
- trying inception model. 
    * ERROR: x sizes: 45, y sizes: 212
    * did the way I augment cause mismatch between x an y?
    * or did I unhash the wrong sections of the model so far. 
    * Managed 1 epoch before error, issue with first epoch output?
========================

### November 5
DONE:
- Created a script ofor image augmentation using imgaug https://github.com/aleju/imgaug
- Created a function to smart crop the images to a given size
- Create algorithm to translate boxes into single class labels

- Ran basic model with ResNet photo size 224x224 pixels
    * some small errors rerunning 224x224
- Also rerunning with 350x350


TODO:
- Create a basic model with resnet & similar to run a classification
- create a preprocessing pipeline

### November 6
DONE:
- ResNet and EfficientNet with KNB dataset
- moved most notebooks and materials to shared Drive folder
- Matteo created a script to automatically log model output and save in Drive
- fought tensorboard
- lost access to Barry's Google bucket
- Lindsey created new Google buscket with ETH photos
    * good for 90 days. ~$270 left in credit after upload

TODO
- run models with more epochs
- Lindsey: 
    * import early stopping and incorporate into model
    * is dense layer in model an issue?

========================

### November 9  

DONE:
- Setup notebook 21 for finetuning and transfer learning of classification models
- found new sources for MASK_R CNN model training
- Ran finetuning notebooks with ResNet and InceptionResNetV2
    * ResNet F1: 0.39
    * InceptionResNetv2 F1: 0.43
    * InceptionResNetv2 + ImgAug: 0.44
    * InceptionResNetv2 + ImgAug + swish/flatten: 0.44
    * InceptionResNetv2 + swish/flatten: 0.42
- Started following MaskRCNN tutorial shared by Barry. 

TODO:
- Define image augmentation procedure for classification models
- Implement fine tuning of MASK-R-CNN using PyTorch
- Try ResNet101 instead of ResNet50
- Fix augmentation bugs, mostly only scoring class 3

### November 10
DONE:
- image augmentation problem comes from improper balancing
- Meeting with Badru and Barry. Suggest:
    * maybe try classification with fewer labels
    * Follow notebook tutorial from Barry
    * Follow Docker from Nitin
    * Finish PyTorch implementation trials
    * Meet with Barry Thursday 10am, Badru Mon 11am
- created seperate JSON files with train/test/val segments
- modified Barry's notebook to work locally
    * the current code doesn't rescale the images. 
    * add rescale step before running model to avoid errors
    * should probably balance as well
- Nitin's Docker 
    * primarily novel segmentation protocol the students put together
    * not very useful directly BUT
    * UNet model would be good to try
    * model architecture has rescaling step built in, making it easier to modify to our dataset

TODO:
- fix balancing issue in ResNet models
    * check function in multiple notebooks
- Run UNet model with ETH images
- add image and segment resizing to Barry's notebook
    * borrow code from the UNet Docker? should work well for our structure
- LINDSEY: modify Jupyter Notebooks and requirement files for GDrive

### November 11
DONE:
- Run UNet model with ETH images and masks
    * problem: masks and images are off by a few pixels
    * when images color: mask predictions cover all the plant
    * when images BW: mask predictions is all background
- Ran second version of UNet (600_UNet _TGS)
    * surprisingly good predicted masks even with offset

TODO:
- is mismatch between ETH images and masks from the files or something Lindsey did?
- test UNet with images/masks from the KNB and CAS datasets
- Lindsey: experiment further with UNet_TGS
     * check with standard ETH images
     * how is it making masks?
     * what exactly is Salt?
- Meeting with Barry 10am tomorrow


### November 12
DONE:
- Meeting with Barry
     * check output channels of models. Should equal number of classification layers (2 or 3)
     * incorporate IOU metrics
     * fix image/mask offset
     * train on CAS and ETH datasets
     * change binary crossentropy (we dont have a binary problem)
- Fixed offset between mask and image
- Use Jaccards_distance as loss function
     * is based on intersection/union
- metrics = [iou] or metrics = [iou, iou_thresholded]

TODO:

    
### November 14-15
- Barry made us obsolete with proper implementation of Mask RCNN form akTwelve git repo

### November 16
TODO:
- fix up mask RCNN models
- create rough presentation
- draft report

DONE:


November 19
- cleaning main README  

### basic data loading and management
helper_function.py : custom functions that are useful across all notebooks.  
requirements.txt : list of required packages and libraries `pip freeze > requirements.txt`  

001_data_loading.ipynb : how to access the data in the google bucket   
001b_data_loading_COLAB : access google bucket using COLAB (failed and partial)
001c_Problems_accessing_data :  collection of screenshot and tries to connect to the damned bucket!!   
002_outline detection: first try at object detection using pre-trained model 
003_ Notebook to visualise all experiments using tensorboard

### Neural networks:
010_first_cnn: first try at mask R_CNN with knb dataset
011_first_object_recognition_model_COLAB: testing of tensorflow hub model for object detection
012_fineTuning_&_transferLearning_COLAB: FAILED attempt to train a  object detection model
013_Model with PyTorch: implemeentation of mask_r_cnn using PyTorch

### Classification
020_finetuning_for_classification_COLAB: preprocessing of knb and first run of classification model using a siomple cnn and models from keras.application
021_finetuning_for_classification_COLAB_clean: finetuning and transfer learning procedure notebook
