import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np


def visual_coco(coco_image, coco_json):
    coco = COCO(coco_json)
    for image_id in tqdm(coco.getImgIds(), desc="visual coco"):
        image_info = coco.loadImgs(ids=image_id)[0]
        image = np.array(Image.open(os.path.join(
            coco_image, image_info['file_name'])))
        plt.imshow(image)
        anno_infos = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        coco.showAnns(anno_infos, draw_bbox=True)
        plt.show()


coco = COCO
if __name__ == "__main__":
    coco_image = ""
    coco_json = ""
    visual_coco(coco_image, coco_json)
