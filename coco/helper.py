import pandas as pd
import numpy as np
from ..helper.file import load_json_file

class COCOHelper():

    def __init__(self, gt_file):
        "
        Params:
            gt_file: Ground truth file path
        "

        self.gt_coco = load_json_file(gt_file)
        self.category_dict = self.get_category_dict()
        self.image_info_dict = self.get_image_info_dict()

    def get_category_dict(self):
        categories = self.gt_coco['categories']
        category_dict = {i['id']: i['name'] for i in categories}
        
        return category_dict

    def get_image_info_dict(self):
        image_infors = self.gt_coco['images']
        image_info_dict = {i['id']: i['file_name'] for i in image_infors}

        return image_info_dict



