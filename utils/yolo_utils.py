import numpy as np
import glob
from PIL import Image, ImageDraw

def plot_bound_box(image, annotation_list):
    """
    Plot image and its bounding box as yolo format
    """

    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h

    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))

        class_name = coco2yolo.get_category_dict()[obj_cls + 1]
        plotted_image.text((x0, y0 - 10), class_name)

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(np.array(image))
    plt.show()
