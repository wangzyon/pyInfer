from typing import List
import numpy as np
import shapely.geometry as shgeo
from .bbox import QuadrangleBBox


def cpu_nms(bboxes: List[QuadrangleBBox], nms_threshold) -> List[QuadrangleBBox]:
    def box_iou(a: QuadrangleBBox, b: QuadrangleBBox):
        a_poly = shgeo.Polygon(np.array(a.coords).reshape(-1, 2).tolist())
        b_poly = shgeo.Polygon(np.array(b.coords).reshape(-1, 2).tolist())
        inter_poly = a_poly.intersection(b_poly)
        if inter_poly.area == 0:
            return 0
        return inter_poly.area / (a_poly.area + b_poly.area - inter_poly.area)

    for i in range(len(bboxes)):
        for j in range(len(bboxes)):
            # label不同或同一个bbox跳过nms
            if i == j or bboxes[i].label != bboxes[j].label:
                continue

            # 置信度相同，保留靠后bbox, 即j<i时不对i进行nms
            if bboxes[j].confidence == bboxes[i].confidence and j < i:
                continue

            if bboxes[j].confidence > bboxes[i].confidence:
                iou = box_iou(bboxes[i], bboxes[j])
                if (iou > nms_threshold):
                    bboxes[i].keepflag = 0

    return list(filter(lambda bbox: bbox.keepflag, bboxes))