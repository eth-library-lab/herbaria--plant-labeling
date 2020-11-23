# -*- coding: utf-8 -*-
""" Script to detect of flowers and fruits in herbarium samples
Allows calling through COmmand Line Interface CLI
"""
# Import modules and packages
import os
import sys
import json
import numpy as np
import csv
import time
from PIL import Image, ImageDraw
import argparse
import matplotlib.pyplot as plt

plt.style.context("fivethirtyeight")
import skimage
from skimage.io import imsave, imread

# import modules from Mask_RCNN
sys.path.append("./src/Mask_RCNN")
print(os.path.abspath("./src/Mask_RCNN"))

try:
    from mrcnn.config import Config
    import mrcnn.utils as utils
    from mrcnn import visualize
    import mrcnn.model as modellib
except ModuleNotFoundError:
    print("modules for MaskRCNN not found, Please check folder path or download mask-RCNN \
        from source: https://github.com/akTwelve/Mask_RCNN")

# import custom model
sys.path.append("./src")
import herbaria as hb


def detect_flowers_fruits(input_images_dir, output_folder, model=None, **kwargs):
    """ Runs predictions on images and save the results in output folder.
    Optionally can:
         Filter the outputs based on the score (confidence of the predictions),
         Save masks as png, Display images and outputs (not through Command Line interface)

    Returns the filename of the images processed correctly and optionally

    Params:
    -input_images_dir: path to folder of input images
    -output_folder: path to output destination folder
    -model : model to use to run predictions (predefined if using CLI)

    OPTIONAL params:
    -save_predictions: BOOL default True
    -save masks: BOOL default False
    -filter_scores : REAL between 0 and 1 default 0.5
    -save stats : BOOL default True saves a csv of detection statistics for each image"""

    save_predictions = kwargs.get("save_predictions", True)
    save_masks = kwargs.get("save_masks", False)
    filter_scores = kwargs.get("filter_scores", 0.5, )
    show_predictions = kwargs.get("show_predictions", False)
    save_stats = kwargs.get("save_stats", True)
    labels = kwargs.get("labels", ['flowers', 'fruits'])

    print(f" running inference with the following options:")
    print(f" image source : ", input_images_dir)
    print(f" destination folder: {output_folder}")
    for k, v in kwargs.items():
        print(k, v)

    if not model:
        model = hb.load_trained_model()

    # checks
    assert os.listdir(input_images_dir), "input folder is empty. Please check path or extension of files"
    if not os.path.exists(output_folder):
        'dest folder does not exist. It will be created ;)'
        os.mkdir(output_folder)

    processed_list = []  # list of processed images
    stats_list = []
    for filename in os.listdir(input_images_dir)[:2]:
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_path = os.path.join(input_images_dir, filename)
        else:
            continue

        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
        results = model.detect([img_arr], verbose=0)  # this is equivalent to .predict() and actually does the inference
        # filter by scores

        r = results[0]  # this contains all the predictions including the masks for each image
        # filter results by confidence score
        if filter_scores > 0:
            print(f"removing predictions with confidence below {filter_scores}")

            sc_ = r['scores']
            lim_ = len(sc_[r['scores'] > filter_scores]) - 1
            r['masks'] = r['masks'][:, :, :lim_]
            r['rois'] = r['rois'][:lim_]
            r['scores'] = r['scores'][:lim_]
            r['class_ids'] = r['class_ids'][:lim_]

        # create stats table for output
        if save_stats:
            # create dictionary with labels
            stats_dic = {
                'img_id': filename.split(".")[0],
                'min_confidence': filter_scores,
                'max_confidence': r['scores'][0],

            }
            for i in labels:
                stats_dic[str(i)+"_count"] = 0
                stats_dic[str(i)+"_area"] = 0
            # add count of flowers and fruits
            labs, counts = np.unique(r['class_ids'], return_counts=True)
            for n in range(len(labs)):
                column_head = labels[labs[n]-1]+"_count"
                # print("column line", column_head)
                stats_dic[column_head] = counts[n]

            # add area of flowers and fruits(from boxes)
            area_sum=[0]*len(labels)
            for n, box in enumerate(r['rois']):
                lab = r['class_ids'][n]
                area = (box[2]-box[0])*(box[3]-box[1])
                area_sum[lab-1] = area_sum[lab-1]+area
            column_head = labels[lab-1] + "_area"
            for n,i in enumerate(labels):
                stats_dic[ i + "_area"] = area_sum[n]
            print(stats_dic)
            stats_list.append(stats_dic)

        # display images
        if show_predictions:
            visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                        labels, figsize=(100, 100))

        # save images
        if save_predictions:
            print(f"\n saving predictions for image {filename} to {output_folder}")
            # create dictionary ( MASKS are EXCLUDED)
            json_pred = {'categories': labels,
                         'image': filename.split(".")[0],
                         'rois': r['rois'].tolist(),
                         'labels': r['class_ids'].tolist(),
                         'scores': r['scores'].tolist()
                         }
            processed_list.append(filename)

            json_path = os.path.join(output_folder, filename.split(".")[0] + ".json")
            with open(os.path.join(json_path), "w+") as file:
                json.dump(json_pred, file)
        # saving masks
        if save_masks:
            # create a single mask with category values
            complete_mask = np.zeros(img_arr.shape[:-1], dtype=np.uint8)
            for n in range(r['masks'].shape[2]):
                single_mask = r['masks'][:, :, n].astype(np.uint8)
                single_mask = single_mask * r['class_ids'][n]
                complete_mask = complete_mask + single_mask
            # save image
            mask_path = os.path.join(output_folder, "MASK-" + filename.split(".")[0] + ".png")
            imsave(mask_path, complete_mask)
            print(f"\n saved predicted masks for image {filename} as {mask_path}")
            processed_list.append(filename)
    # save stats as csv
    if save_stats:
        stat_path = os.path.join(output_folder, "detection_stats.csv")
        keys = stats_list[0].keys()
        with open(stat_path, 'w+', newline="")as output_file :
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats_list)
    proc_images_n = len(set(processed_list))
    print("processing finished !!")
    print(f"{proc_images_n} images elaborated")
    return set(processed_list)


if __name__ == "__main__":
    print("""script to preprocess image for the brassicas detection model. \
    Runs predictions on images and save the results in output in folder.
    Optionally can:
         Filter the outputs based on the score (confidence of the predictions),
         Save masks as png,
         Display images and outputs (not through Command Line interface)

    Returns the filename of the images processed correctly and optionally

    Params:
    -input: abs path to folder of input images
    -output: abs path to output destination folder
    -model : model to use to run predictions (automatically loaded for CLI) 

    OPTIONAL params:
    -sp: save predictions (boxes+labels) as json files BOOL default True
    -save masks: save masks BOOL default False
    -fs : filter results based on confidence scores - REAL between 0 default 0.5
    """)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="location of input images to be processed")
    parser.add_argument("--output", help="location for detection_files")

    # optional arguments
    parser.add_argument("--sp", help=" OPTIONAL save predicted boxes and labels to json file in output folder",
                        default=False, action='store_true')
    parser.add_argument("--sm", help=" OPTIONAL save predicted boxes and labels to json file in output folder",
                        default=False, action='store_true')
    parser.add_argument("--fs", help="filter results based on confidence scores REAL default 0.5", default=0.5)
    parser.add_argument("--st", help="save detection stats as csv file", default=True, action='store_true')
    parser.add_argument("--lb", help="list of labels ", default=['flowers', 'fruits'])
    args = parser.parse_args()


    detect_flowers_fruits(args.input,
                          args.output,
                          model=None,
                          save_predictions=args.sp,
                          save_masks=args.sm,
                          filter_scores=args.fs,
                          save_stats=args.st,
                          labels=args.lb)

