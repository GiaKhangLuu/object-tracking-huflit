import pandas as pd
import numpy as np
import tqdm
import sys
import os

# Root directory of the projet
ROOT_DIR = os.path.abspath('../')
# Find local package
sys.path.append(ROOT_DIR)

from helper.file import load_json_file

class COCOHelper():

    def __init__(self, gt_file):
        """
        Params:
            gt_file: Ground truth file path
        """

        self.annot_coco = load_json_file(gt_file)
        self.category_dict = self.get_category_dict()
        self.image_name_dict = self.get_image_name_dict()

    def get_category_dict(self):
        """
        Get category_dict: `key` is category_id (start from 1)
                           `value` is category_name
        Return:
            {category_id: category_name, ... }
        """
        categories = self.annot_coco['categories']
        category_dict = {i['id']: i['name'] for i in categories}
        
        return category_dict

    def get_image_name_dict(self):
        """
        Get image_name_dict: `key` is image_id {start from 1}
                             `value` is image_name
        Return:
            {image_id: image_name, ... }
        """
        image_name = self.annot_coco['images']
        image_name_dict = {i['id']: i['file_name'] for i in image_name}

        return image_name_dict

class COCO2YOLO(COCOHelper):

    def compute_yolo_box(self, bbox, img_size):
        """
        1. Compute x_center, y_center by formula:
            x_center = (width / 2) + x1 - 0.5
            y_center = (height / 2) + y1 - 0.5
        2. Normalize x_center, y_center, w, h:
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
        Appendix:
            width = x2 - x1 + 1
            => x2 = width + x1 - 1
            => x_center = (width / 2) + x1 - 0.5
        Params:
            bbox: list [x1, y1, w, h]
            img_size: tuple (img_width, img_height)
        Return:
            bbox: [x_center_norm, y_center_norm, 
                   width_norm, height_norm]
        """ 

        x1, y1, w, h = bbox           
        img_w, img_h = img_size

        # Compute x_center, y_center            
        x_center = (w / 2) + x1 - 0.5
        y_center = (h / 2) + y1 - 0.5

        # Normalize
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        return [x_center_norm, y_center_norm, w_norm, h_norm]

    def convert_coco_box_to_yolo_box(self):
        """
        Convert
        Get YOLO formatted annotation 
        Return:
        """

        images = pd.DataFrame.from_records(self.annot_coco['images'])
        annots = pd.DataFrame.from_records(self.annot_coco['annotations'])

        annots = (annots.groupby('image_id')
                        ['category_id', 'bbox'].agg(list))
        images = images[['id', 'file_name', 'width', 'height']]

        images_have_annot = images.set_index('id').join(annots, 
                                                        how='right')
        images_no_annot = images[~images['file_name'].isin(images_have_annot.index)]

        def conver(self, r):
            img_size = (r['width'], r['height'])
            bboxes = r['bbox']
            yolo_bboxes = [self.compute_yolo_box(bbox, img_size) 
                           for bbox in bboxes]
            return yolo_bboxes

        bbox_yolo = images_have_annot.apply(self.convert, axis=1)
        category_id = images_have_annot.apply(
                lambda cat_ids: [cat_id - 1 for cat_id in cat_ids])

        images_have_annot[['category_id', 'bbox_yolo']] = [category_id, bbox_yolo]

        return images_have_annot, images_no_annot









     
