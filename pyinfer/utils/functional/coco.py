import json
from pycocotools.coco import COCO


class COCOCreator():
    def __init__(self) -> None:
        self.image_id_to_index = {}
        self.anno_id_to_index = {}
        self.cat_id_to_index = {}
        self.dataset = {"images": [], "annotations": [], "categories": []}

    def create_image_info(self, image_id, file_name, height, width, update=False, **kwargs):
        image_info = {"id": image_id, "file_name": file_name, "width": width, "height": height, **kwargs}
        if update:
            self.update_image_info(image_info)
        return image_info

    def create_anno_info(self,
                         image_id,
                         anno_id,
                         category_id,
                         bbox,
                         area,
                         segmentation=[],
                         iscrowd=0,
                         update=False,
                         **kwargs):
        anno_info = {
            "image_id": image_id,
            "id": anno_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
            **kwargs
        }
        if update:
            self.update_anno_info(anno_info)
        return anno_info

    def create_cat_info(self, cat_id, cat_name, update=False, **kwargs):
        cat_info = {"id": cat_id, "name": cat_name, **kwargs}
        if update:
            self.update_cat_info(cat_info)
        return cat_info

    def update_image_info(self, image_info):
        image_id = image_info["id"]
        if image_id in self.image_id_to_index:
            self.dataset["images"][self.image_id_to_index[image_id]].update(image_info)
        else:
            self.image_id_to_index[image_id] = len(self.dataset["images"])
            self.dataset["images"].append(image_info)

    def update_anno_info(self, anno_info):
        anno_id = anno_info["id"]
        if anno_id in self.anno_id_to_index:
            self.dataset["annotations"][self.anno_id_to_index[anno_id]].update(anno_info)
        else:
            self.anno_id_to_index[anno_id] = len(self.dataset["annotations"])
            self.dataset["annotations"].append(anno_info)

    def update_cat_info(self, cat_info):
        cat_id = cat_info["id"]
        if cat_id in self.cat_id_to_index:
            self.dataset["categories"][self.cat_id_to_index[cat_id]].update(cat_info)
        else:
            self.cat_id_to_index[cat_id] = len(self.dataset["categories"])
            self.dataset["categories"].append(cat_info)

    def build(self):
        coco = COCO()
        coco.showAnns()
        coco.dataset = self.dataset
        coco.createIndex()
        return coco

    def write(self, path):
        with open(path, 'w') as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)
