import json
import pandas as pd
import numpy as np
import os
import sys

# Root directory of the projet
ROOT_DIR = os.path.abspath('../')
# Find local package
sys.path.append(ROOT_DIR)

from helper.file import load_json_file, write_to_json_file
from coco.helper import COCOHelper

def convert_coco_pred_to_bdd100k(pred_file, gt_file, result_file):
    """ 
    To be able to evaluate the prediction by BDD100K,
    the prediction must be in the correct format. Converting 
    predictions in COCO format to BDD100K format.
    
    Check out the correct format: https://doc.bdd100k.com/evaluate.html

    Params:
        pred_file: Json file contains the predictions
        gt_file: Json file contains the ground truths
        result_file: Json file contains the predictions in BDD100K format
    """
    
    coco_helper = COCOHelper(gt_file)
    pred_coco_format = load_json_file(pred_file) 
    pred_coco_df = pd.DataFrame.from_records(pred_coco_format)

    # Create box2d_x1y1x2y2 column, convert by the equation
    # x2 = x1 + width
    # y2 = y1 + height
    box2d_x1y1x2y2 = pred_coco_df['bbox'].apply(
        lambda r: {'x1': r[0], 
                   'y1': r[1], 
                   'x2': r[0] + r[2], 
                   'y2': r[1] + r[3]
                   }
    )

    # Create category_name column
    category_names = pred_coco_df['category_id'].apply(
        lambda r: coco_helper.category_dict[r]
    )

    # Create image_name column         
    image_names = pred_coco_df['image_id'].apply(
        lambda r: coco_helper.image_info_dict[r]
    )

    # Create prediction_id column
    pred_id = range(1, len(pred_coco_df) + 1)

    pred_coco_df = pred_coco_df.assign(
        box2d_x1y1x2y2 = box2d_x1y1x2y2,
        category_name = category_names,
        name = image_names,
        pred_id = pred_id
    ) 

    # Convert pred_id to string
    pred_coco_df['pred_id'] = pred_coco_df['pred_id'].astype(str)
   
    # Group df by image name 
    pred_groupby_name = (pred_coco_df.
                         groupby(['name'])
                         [['score', 'box2d_x1y1x2y2', 
                           'category_name', 'pred_id']])
    pred_groupby_name = pred_groupby_name.agg(list).reset_index()

    def convert(image):
        pred_of_image = zip(image['pred_id'], 
                            image['category_name'], 
                            image['score'], 
                            image['box2d_x1y1x2y2'])

        return [{'id': i[0], 
                 'category': i[1], 
                 'score': i[2], 
                 'box2d': i[3]} for i in x]

    # Get label column, this column is the list of all
    # predictions per image
    label = pred_groupby_name.apply(convert, axis=1)
    pred_groupby_name['labels'] = label

    pred_bdd100k_format = pred_groupby_name[['name', 'labels']].to_dict('records')

    # Write to json file
    result_file = write_to_json_file(pred_bdd100k_format, result_file)    

    return result_file




