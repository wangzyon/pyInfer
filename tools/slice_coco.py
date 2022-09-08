from pyinfer.utils.common.slice import slice_coco_dataset

if __name__ == "__main__":
    coco_image_src = "C:/Users/wzy/Desktop/xxx/async_infer_python/workspace/coco/train"
    coco_image_dst = "C:/Users/wzy/Desktop/xxx/async_infer_python/workspace/coco/slice_train"

    coco_json_src = "C:/Users/wzy/Desktop/xxx/async_infer_python/workspace/coco/train.json"
    coco_json_dst = "C:/Users/wzy/Desktop/xxx/async_infer_python/workspace/coco/slice_train.json"

    slice_coco_dataset(
        coco_image_src,
        coco_image_dst,
        coco_json_src,
        coco_json_dst,
        rate=1,
        subsize=640,
        gap=200,
        padding=True,
        threshold=0.2)
