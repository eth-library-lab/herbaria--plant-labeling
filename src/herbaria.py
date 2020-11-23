"""
Customised classes and functions for the herbaria classification project
Based on the following repo: https://github.com/akTwelve/Mask_RCNN by Waleed Abdulla

coded by Matteo Jucker Riva (https://github.com/ciskoh) and Lindsey Parkinson (https://github.com/LVParkinson/LVParkinson.github.io)
Thanks to Barry Sunderland (https://github.com/BarrySunderland) for the help !!!

Release under CC li9cense: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import sys
import json
import numpy as np



ROOT_DIR = os.path.join(os.path.dirname(__file__),"Mask_RCNN")

# assert os.path.exists(ROOT_DIR), 'Mask_RCNN module not found.'
sys.path.append(ROOT_DIR)  # To find local version of the library

sys.path.append(ROOT_DIR)
try:
    from mrcnn.config import Config
    import mrcnn.utils as utils
    from mrcnn import visualize
    import mrcnn.model as modellib
except ModuleNotFoundError:
    print(
        "modules for MaskRCNN not found, Please check folder path or download mask-RCNN from source: https://github.com/akTwelve/Mask_RCNN by Waleed Abdulla")


# **********************************************************************************************
#             Classes for custom Mask_RCNN model 
# **********************************************************************************************
class HerbariaConfig(Config):
    """Configuration for training on the eth herbaria dataset.
    Derives from the base Config class of mrcnn and overrides values specific
    to the brassica dataset.
    """
    # Give the configuration a recognizable name
    NAME = "M_image_augm"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 [ 'flower', fruit]

    # All of our training images are 1024x1024
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 4

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 1

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    LEARNING_RATE = 0.01


class InferenceConfig(HerbariaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.5


class HerbariaDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


# **********************************************************************************************
#             Helper functions 
# **********************************************************************************************

def load_trained_model(model_path="model_weights/m_image_augm20201116T0937/mask_rcnn_m_image_augm_0009.h5",
                       mode="inference", config=None):
    """function to load a model with pre_trained weights.
    Params:
    - model_path : STRING path saved weights
    - mode : STRING "inference" or "train" DEFAULT "inference"
    - config :  object of class Config, defaults to InferenceConfig for inference and HerbariaConfig for training
    Returns a pre-trained model ready for training or inference"""
    abs_path = os.path.dirname(__file__)
    model_rel_path=model_path.split("/")
    print(f"loading weights from {model_path}")

    if config == None:
        if mode == 'train':
            config = HerbariaConfig()
        elif mode == 'inference':
            config = InferenceConfig()
        else:
            assert " 'mode' should be 'train' or 'inference' "

    print("Loading weights from ", model_path)
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir="./model_weights")
    # load weights
    model.load_weights(model_path, by_name=True)
    return model

if __name__ == "__main__":
    print(" Main module for herbaria classification model.")
    print("\n testing HerbariaConfigClass")
    config=HerbariaConfig()
    config.display()
    print("\n testing load_trained_model function")
    load_trained_model()

