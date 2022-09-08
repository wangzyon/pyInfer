import cv2
import copy
import math
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import shapely.geometry as shgeo
from typing import List
from pycocotools.coco import COCO

from ..detection.bbox import QuadrangleBBox
from .coco import COCOCreator

__all__ = [""]


def reduce_coords(coords):
    """将最短边的两个点p1,p2，用p1,p2中点p替换，从而使多边形减少一个顶点"""
    def get_distance(point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    coords.append(coords[0])
    distances = [get_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]

    min_index = np.array(distances).argsort()[0]
    index = 0
    out_coords = []
    while index < len(coords):
        if (index == min_index):  # 取中点
            middle_x = (coords[index][0] + coords[index + 1][0]) / 2
            middle_y = (coords[index][1] + coords[index + 1][1]) / 2
            out_coords.append((middle_x, middle_y))
        elif (index != (min_index + 1)):  # 非最短边点保留
            out_coords.append((coords[index][0], coords[index][1]))
        index += 1
    return out_coords


def choose_best_pointorder_fit_another(coords, gt_coords):
    """
    :params coords: [p1,p2,p3,p4], 四边形坐标点
    :params gt_coords: [p1,p2,p3,p4]， 标签四边形坐标点
    
    选择最匹配gt_coords的坐标排布顺序，coords在最优排布顺序下，与gt_coords逐点点间距之和最小；
    """
    x1, gt_x1 = coords[0][0], gt_coords[0][0]
    y1, gt_y1 = coords[0][1], gt_coords[0][1]
    x2, gt_x2 = coords[1][0], gt_coords[1][0]
    y2, gt_y2 = coords[1][1], gt_coords[1][1]
    x3, gt_x3 = coords[2][0], gt_coords[2][0]
    y3, gt_y3 = coords[2][1], gt_coords[2][1]
    x4, gt_x4 = coords[3][0], gt_coords[3][0]
    y4, gt_y4 = coords[3][1], gt_coords[3][1]

    combinate = [
        np.array([x1, y1, x2, y2, x3, y3, x4, y4]),
        np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
        np.array([x3, y3, x4, y4, x1, y1, x2, y2]),
        np.array([x4, y4, x1, y1, x2, y2, x3, y3])
    ]
    gt = np.array([gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4])
    distances = np.array([np.sum((coord - gt)**2) for coord in combinate])
    sorted = distances.argsort()
    best_coords = combinate[sorted[0]].reshape(-1, 2).tolist()
    return best_coords


def assign_bboxes(pleft, ptop, pright, pbottom, gt_bboxes: List[QuadrangleBBox], threshold):
    """
    分配图像bounding box，过滤掉坐标不在当前图像内的bbox

    Args:
        pleft, ptop, pright, pbottom: 图像左上角和右下角坐标值
        bboxes (List[QuadrangleBBox], optional): 图像上目标的bounding box. Defaults to None.
        threshold (float, optional): 目标bbox和瓦片重叠IOU阈值，大于阈值将bbox分配给该瓦片，否则丢弃. Defaults to 0.5.

    Returns:
        patches: List[bboxes]
        
    基本步骤：
    
    1. 遍历一个gt_bbox;
    2. 计算bbox和当前图像的iou;
    3. iou=1, bbox完全在图像内，保留，跳至第一步；
    4. iou>threshold， bbox部分在图像内，重叠多边形顶点数为n
        ① n<4, bbox只有一个顶点在图像内，bbox不匹配；
        ② n=5，将5边形变换为4边形，生成新的new_bbox，匹配；
        ③ n=6，将6边形变换为4边形，生成新的new_bbox，匹配；
        ④ n>6,实际使用中，该情况较少发生，不匹配；
    5. 调整new_bbox顶点顺序，即new_bbox以第p个顶点作为起始点，该顶点排布顺序，与gt_bbox对应顶点距离和最小，
    6. new_bbox坐标值是相对原点(0,0)的值，以图像左上角点为原点，更新坐标值；
    7. 重复第一步;
    """
    image_poly = shgeo.Polygon([(pleft, ptop), (pright, ptop), (pright, pbottom), (pleft, pbottom)])

    remain_bboxes = []
    for gt_bbox in gt_bboxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = gt_bbox.coords
        gt_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        gt_poly = shgeo.Polygon(gt_coords)
        # 错误的bbox
        if (gt_poly.area <= 0):
            continue

        inter_poly = gt_poly.intersection(image_poly)  # 重叠区域
        iou = inter_poly.area / gt_poly.area  # 非常规IOU，此IOU时交集与gt_box的比值

        # 若bbox在图像内部，则保留，同时以图像的左上角点作为bbox原点
        if (iou == 1):
            bbox = copy.deepcopy(gt_bbox)  # 更新原点,不要在原始bbox上直接修改
            remain_bboxes.append(bbox.reset_origin(pleft, ptop))

        # 若bbox部分在图像内部
        elif iou > threshold:
            # 有序排布多边形点；
            inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
            # 重叠区域多边形顶点，由于收尾相接，第一个点和最后一个点重复，因此去除最后一个点；
            # inter_coords: [(x1, y1), (x2, y2), (x3, y3), (x4, y4), ..., (x1, y1)]
            inter_coords = list(inter_poly.exterior.coords)[0:-1]

            # 两个矩形重叠区域较小时，重叠区可能由三个点组成，重叠较小，bbox不保留
            if len(inter_coords) < 4:
                continue

            elif (len(inter_coords) > 6):
                """两个矩形重叠区域存在6个以上顶点，在实际情况中极少出现，因此此类bbox做丢弃处理"""
                continue

            elif (len(inter_coords) == 6):
                """重叠区域6个顶点时，合并两个最短边，减少顶点数至4"""
                inter_coords = reduce_coords(reduce_coords(inter_coords))

            elif (len(inter_coords) == 5):
                """重叠区域5个顶点时，合并一个最短边，减少顶点数至4"""
                inter_coords = reduce_coords(inter_coords)

            # elif (len(inter_coords)==4):
            #     4个顶点不作处理

            best_coords = choose_best_pointorder_fit_another(inter_coords, gt_coords)  # 最优顶点顺序
            best_coords_area = shgeo.Polygon(best_coords).area
            # 限制坐标范围, 并更新原点
            best_coords = np.array(best_coords).flatten()
            new_bbox = QuadrangleBBox(label=gt_bbox.label, area=best_coords_area)
            new_bbox.coords = best_coords
            new_bbox.clip(xmin=pleft, xmax=pright, ymin=ptop, ymax=pbottom)
            new_bbox.reset_origin(ox=pleft, oy=ptop)
            remain_bboxes.append(new_bbox)
    return remain_bboxes


def slice_patch(image, left, top, subsize, padding):
    no_padding_patch = copy.deepcopy(image[top:(top + subsize), left:(left + subsize)])
    h, w, c = no_padding_patch.shape
    if (padding):
        patch_image = np.ones((subsize, subsize, c)) * 114
        patch_image[0:h, 0:w, :] = no_padding_patch
    else:
        patch_image = no_padding_patch
    return patch_image.astype(np.uint8)


def slice_one_image(image: np.ndarray,
                    image_base_name,
                    subsize,
                    rate,
                    gap,
                    threshold=0.5,
                    padding=True,
                    bboxes: List[QuadrangleBBox] = None):
    """
    拆分一张图像

    Args:
        image (np.ndarray): 图像
        subsize (int): 瓦片大小
        rate (int): 图像resize倍数, rate=1表示不对图像做resize处理
        gap (int): 瓦片重叠大小
        bboxes (List[QuadrangleBBox], optional): 图像上目标的bounding box. Defaults to None.
        threshold (float, optional): 目标bbox和瓦片重叠IOU阈值，大于阈值将bbox分配给该瓦片，否则丢弃. Defaults to 0.5.
        padding (bool, optional): 瓦片大小不足subsize，是否padding. Defaults to True.
        name (str, optional): 图像名称. Defaults to "".

    Returns:
        patches: List[Tuple[patch_name, patch_image, bboxes]]
        
    基本步骤：
    
    1. 选取瓦片图像范围
    2. 从image切分瓦片patch image
    3. 若image有标签，将标签bboxes与瓦片进行匹配，匹配成功的部分bboxes作为patch bboxes
    4. 滑动，重复步骤1~3
    """
    assert np.shape(image) != ()

    patches = []
    if (rate != 1):
        if bboxes is not None:
            bboxes = list(map(lambda bbox: bbox * rate, bboxes))
        resizeimg = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    else:
        resizeimg = image

    base_name = image_base_name + '__' + str(rate) + '__'

    width = np.shape(resizeimg)[1]
    height = np.shape(resizeimg)[0]

    slide = subsize - gap
    left, top = 0, 0
    while (left < width):
        if (left + subsize >= width):
            left = max(width - subsize, 0)
        top = 0
        while (top < height):
            if (top + subsize >= height):
                top = max(height - subsize, 0)
            patch_name = base_name + str(left) + '___' + str(top)
            right = min(left + subsize - 1, width - 1)
            bottom = min(top + subsize - 1, height - 1)

            # 获取一张瓦片图
            patch_image = slice_patch(resizeimg, left, top, subsize, padding)

            # 获取瓦片图内的bbox
            if bboxes is None:
                patch_bboxes = []
            else:
                patch_bboxes = assign_bboxes(left, top, right, bottom, bboxes, threshold)
            patch = (patch_name, rate, left, top, patch_image, patch_bboxes)
            patches.append(patch)

            # 滑动
            if (top + subsize >= height):
                break
            else:
                top = top + slide
        if (left + subsize >= width):
            break
        else:
            left = left + slide
    return patches


def slice_coco_dataset(coco_image_src,
                       coco_image_dst,
                       coco_json_src,
                       coco_json_dst,
                       rate,
                       subsize,
                       gap,
                       padding=True,
                       threshold=0.5):
    coco = COCO(coco_json_src)
    coco_creator = COCOCreator()

    image_cnt = 0
    anno_cnt = 0
    for parent_image_id in tqdm(coco.getImgIds(), desc="slice"):
        # 1.读取图像
        image_info = coco.loadImgs(ids=parent_image_id)[0]
        image = np.array(Image.open(os.path.join(coco_image_src, image_info['file_name'])))
        image_base_name, extension = os.path.splitext(image_info['file_name'])

        # 2.获取图像gt_bboxes
        anno_infos = coco.loadAnns(coco.getAnnIds(imgIds=parent_image_id))
        gt_bboxes = []
        for ann_info in anno_infos:
            left, top, width, height = ann_info['bbox']
            right, bottom = left + width, top + height
            gt_bbox = QuadrangleBBox(x1=left,
                                     y1=top,
                                     x2=right,
                                     y2=top,
                                     x3=right,
                                     y3=bottom,
                                     x4=left,
                                     y4=bottom,
                                     label=ann_info['category_id'],
                                     area=ann_info['area'])
            gt_bboxes.append(gt_bbox)

        # 3.图像切片，gt_bboxes分配
        patches = slice_one_image(image=image,
                                  image_base_name=image_base_name,
                                  subsize=subsize,
                                  gap=gap,
                                  bboxes=gt_bboxes,
                                  rate=rate,
                                  threshold=threshold,
                                  padding=padding)

        for patch_name, rate, left, top, patch_image, patch_bboxes in patches:
            if len(patch_bboxes) == 0:  # 无目标的切片丢弃
                continue

            patch_image_filename = f"{patch_name}{extension}"

            # 4 保存切片图像
            Image.fromarray(patch_image.astype(np.uint8)).save(os.path.join(coco_image_dst, patch_image_filename))

            # 5 新增切片的image_info
            image_info = coco_creator.create_image_info(
                image_id=image_cnt,
                file_name=patch_image_filename,
                height=subsize,
                width=subsize,
                parent_image_id=parent_image_id,  # 来自同一大图的切片具有相同的大图id
                slice_params=[rate, left, top])  # 切片左上角在大图中的坐标，大图切片前的resize参数rate
            coco_creator.update_image_info(image_info)

            # 6 新增切片的anno_info
            for bbox in patch_bboxes:
                bbox: QuadrangleBBox
                anno_info = coco_creator.create_anno_info(image_id=image_cnt,
                                                          anno_id=anno_cnt,
                                                          bbox=[bbox.left, bbox.top, bbox.width, bbox.height],
                                                          category_id=bbox.label,
                                                          area=bbox.area)
                coco_creator.update_anno_info(anno_info)
                anno_cnt += 1
            image_cnt += 1

        # 直接引用大图的cat_info
        for cat_info in coco.loadCats(coco.getCatIds()):
            coco_creator.update_cat_info(cat_info)

        # 生成新的coco_json
        coco_creator.write(coco_json_dst)