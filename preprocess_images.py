# -*- coding: utf-8 -*-
"""Script to preprocess images for training & detection of flowers and fruits in herbarium samples

# image and annotation processing
resizes images
scales annotations appropriately
removes segments that have too few points
"""

import json
import os
import argparse
import numpy as np
from PIL import Image


def calc_resize_with_apect(size, max_dimension):
    """
    calculate new image size that preserves the aspect ratio but does not exceed the max dimension
    size: tuple, current image size
    max_dimension: int, dimension not to exceed
    """
    w = size[0]
    h = size[1]

    # if min(size) > min_dimension:

    max_orig_dim = max(size)

    new_w = (w / max_orig_dim) * max_dimension
    new_h = (h / max_orig_dim) * max_dimension

    new_size = (int(new_w), int(new_h))

    return new_size


def resize_image(pil_image, max_dimension, return_ratio=False):
    """
    resize a pil image to have the maximum dimension given on oneside while preserving aspect ratio
    pil_image: PIL.Image, image to resize
    max_dimension: int, max size
    return_ratio: bool, return the original image size divided by new image size
    
    """

    orig_size = pil_image.size
    new_size = calc_resize_with_apect(orig_size, max_dimension)
    pil_image = pil_image.resize(new_size, resample=Image.ANTIALIAS)

    if return_ratio:
        resized_ratio = orig_size[0] / new_size[0]
        return pil_image, resized_ratio

    return pil_image


def pad_image_to_square(pil_image, constant_values=0):
    """
        convert a rectangular image to a square by padding the smaller side with a constant value
        pil_image: PIL.Image, 
        constant_values: int, 0-255 value to pad image with 
        """

    im_array = np.array(pil_image)

    h = im_array.shape[0]
    w = im_array.shape[1]

    h_padding = max(0, (w - h))
    w_padding = max(0, (h - w))

    h_padding_bottom = np.ceil(h_padding).astype(int)
    w_padding_bottom = np.ceil(w_padding).astype(int)

    paddings = np.array([[0, h_padding_bottom],
                         [0, w_padding_bottom],
                         [0, 0]])

    im_array = np.pad(im_array, paddings, constant_values=constant_values)

    return Image.fromarray(im_array)


def resize_pad_and_save_image(input_fpath, img_output_fldr_path, max_dimension=1024):
    im = Image.open(input_fpath)
    # this paramater can be returned by the resize image function, 
    # in order to also scale the annotations
    resized_ratio = None
    im, resized_ratio = resize_image(im, max_dimension, return_ratio=True)
    im_s = pad_image_to_square(im)
    # save resized image
    fname = os.path.basename(input_fpath)
    output_fpath = os.path.join(img_output_fldr_path, fname)
    im_s.save(output_fpath)

    if resized_ratio:
        return resized_ratio


def process_images_and_annotations(img_src_fldr_path, img_output_fldr_path, orig_anno_dict=None, split='train',
                                   max_img_dimension=1024):
    missing_files = []
    ratios_dict = {}
    new_annotations = []

    # make sure output folder exists
    if not split:
        split = 'train'
    img_output_fldr_path = os.path.join(img_output_fldr_path, split)
    if not os.path.exists(img_output_fldr_path):
        os.makedirs(img_output_fldr_path)
        print('created directory: {}'.format(img_output_fldr_path))

    # Process Images
    print('outputting images to: {}'.format(img_output_fldr_path))
    if not orig_anno_dict:
        img_list = [a for a in os.listdir(img_src_fldr_path) if a.split(".")[-1].lower() in ["jpg", "png", "tiff"]]
        print(os.listdir(img_src_fldr_path))
        print(img_list)
        for record, fname in enumerate(img_list):
            print(f"processing image {fname}")
            src_path = os.path.join(img_src_fldr_path, fname)

            if os.path.exists(src_path):
                resized_ratio = resize_pad_and_save_image(src_path, img_output_fldr_path,
                                                          max_dimension=max_img_dimension)
                # in case all pictures are not the same size, keep a dict of image id and resized ratios,
                # to resize the annotations later
                ratios_dict[record] = resized_ratio

            else:
                missing_files.append(record)
                print('no src file: ', fname)

    # Process annotations
    if orig_anno_dict:
        for record in orig_anno_dict['images']:

            fname = record['file_name']
            src_path = os.path.join(img_src_fldr_path, fname)

            if os.path.exists(src_path):
                resized_ratio = resize_pad_and_save_image(src_path, img_output_fldr_path,
                                                          max_dimension=max_img_dimension)
                # in case all pictures are not the same size, keep a dict of image id and resized ratios,
                # to resize the annotations later
                ratios_dict[record['id']] = resized_ratio

            else:
                missing_files.append(record['id'])
                print('no src file: ', record['file_name'])

        for orig_anno in orig_anno_dict['annotations']:

            if orig_anno['image_id'] in missing_files:
                continue

            image_id = orig_anno['image_id']
            new_anno = orig_anno

            # scale segmentation
            try:
                resize_ratio = ratios_dict[image_id]
            except:
                print("annotation was skipped for image_id: {}".format(image_id))
                next

            scaled_segmentations = []
            for seg in orig_anno['segmentation']:
                # remove segments with less than 6 points as this can't make a mask
                if len(seg) < 6:
                    print('less than 6 points in segment for annotation: ', orig_anno['id'], ' image_id: ', image_id)
                    continue

                scaled_segmentation = np.array(seg) / resize_ratio
                scaled_segmentation = np.round(scaled_segmentation, decimals=1).tolist()
                scaled_segmentations.append(scaled_segmentation)

            new_anno['segmentation'] = scaled_segmentations
            # scale bounding box
            new_anno['bbox'] = [np.round(pnt / resize_ratio, 1) for pnt in orig_anno['bbox']]
            # scale area
            new_anno['area'] = np.round((orig_anno['area'] / resize_ratio ** 2))
            # images_sizes
            new_anno['width'] = max_img_dimension
            new_anno['height'] = max_img_dimension

            new_annotations.append(new_anno)

        new_image_records = []

        for img_record in orig_anno_dict['images']:
            if img_record['id'] in missing_files:
                continue
            img_record['path'] = os.path.join(img_output_fldr_path, img_record['file_name'])
            img_record['width'] = max_img_dimension
            img_record['height'] = max_img_dimension
            new_image_records.append(img_record)

        # save new annotation json file
        new_dict = {'categories': orig_anno_dict['categories'], 'annotations': new_annotations,
                    'images': new_image_records}

        output_fpath = os.path.join(img_output_fldr_path, split + '.json')
        with open(output_fpath, 'w') as f:
            json.dump(new_dict, f)

        return new_dict


def load_annotations_dict(fpath, split='train'):
    # files_d = {'test': 'Brassicas_Train_Segments.json', 'train': 'Brassicas_Test_Segments.json'}
    # fpath = os.path.join(orig_anno_fldr_path, files_d[split])

    with open(fpath, 'r') as f:
        orig_anno_dict = json.load(f)

    return orig_anno_dict


"""# Main"""

# set folder paths
# orig_anno_fldr_path = os.path.join(project_dir,"ETH_DATASET", )
# img_src_fldr_path = os.path.join(project_dir,"ETH_DATASET",'segmented_brassicas')
# img_output_fldr_path = os.path.join(project_dir,"ETH_DATASET",'preproc_2048')

# load original annotations
"""split='train'

orig_anno_dict = load_annotations_dict(orig_anno_fldr_path, split=split)
# process images and annotatons
new_anno_dict_train = process_images_and_annotations(img_src_fldr_path, 
                               img_output_fldr_path, 
                               orig_anno_dict, 
                               split=split, 
                               max_img_dimension=2048)"""


# load original annotations
def prepare_sample(img_src_fldr_path, img_output_fldr_path, max_dim, orig_anno_fldr_path=None, split=None):
    if orig_anno_fldr_path:
        orig_anno_dict = load_annotations_dict(orig_anno_fldr_path, split=split)
    else:
        orig_anno_dict = None
    # process images and annotatons
    new_anno_dict = process_images_and_annotations(img_src_fldr_path,
                                                   img_output_fldr_path,
                                                   orig_anno_dict,
                                                   split=split,
                                                   max_img_dimension=max_dim)
    print("processing finished")


if __name__ == "__main__":
    print("script to preprocess image for the brassicas detection model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="location of input images to be processed")
    parser.add_argument("--output", help="location for processed images")
    parser.add_argument("--maxdim", help=" max dimension of output images", type=int)
    parser.add_argument("--annot", help=" OPTIONAL path to json file with annotations in COCO format")
    parser.add_argument("--split", help="OPTIONAL train or test split")
    args = parser.parse_args()

    print(args)
    prepare_sample(args.input, args.output, args.maxdim, args.annot, args.split)
