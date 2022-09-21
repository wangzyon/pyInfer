import cv2
import numpy as np

__all__ = ["WarpAffineTraits"]


class WarpAffineTraits():
    "仿射变换"

    def __init__(self, sx, sy, dx, dy):
        self.sx = sx
        self.sy = sy
        self.dx = dx
        self.dy = dy
        # 仿射变换矩阵
        self.m2x3_to_dst, self.m2x3_to_src = self.init()

    def init(self):
        scale_x = self.dx / self.sx
        scale_y = self.dy / self.sy
        self.scale = min(scale_x, scale_y)

        self.tx = round(self.scale * self.sx)
        self.ty = round(self.scale * self.sy)

        # keep ratio resize
        m2x3_to_dst = np.zeros((2, 3))
        m2x3_to_dst[0][0] = self.scale
        m2x3_to_dst[0][1] = 0
        m2x3_to_dst[0][2] = -self.scale * self.sx * 0.5 + self.dx * 0.5 + self.scale * 0.5 - 0.5
        m2x3_to_dst[1][0] = 0
        m2x3_to_dst[1][1] = self.scale
        m2x3_to_dst[1][2] = -self.scale * self.sy * 0.5 + self.dy * 0.5 + self.scale * 0.5 - 0.5
        m2x3_to_dst = m2x3_to_dst.astype(np.float32)

        m2x3_to_src = cv2.invertAffineTransform(m2x3_to_dst).astype(np.float32)
        return m2x3_to_dst, m2x3_to_src

    def __call__(self, src_img: np.ndarray, interpolation=cv2.INTER_LINEAR, pad_value=[114, 114, 114]):
        "对输入图像进行仿射变换"
        top = int((self.dy - self.ty) * 0.5)
        left = int((self.dx - self.tx) * 0.5)
        bottom = self.dy - self.ty - top
        right = self.dx - self.tx - left
        dst_img = cv2.resize(src_img, (0, 0), fx=self.scale, fy=self.scale, interpolation=interpolation)
        dst_img = cv2.copyMakeBorder(dst_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
        return dst_img

    def to_src_coord(self, dx, dy):
        "变换后坐标->变换前坐标"
        sx, sy = np.matmul(self.m2x3_to_src, np.array([dx, dy, 1]).T)
        sx = min(max(round(sx), 0), self.sx - 1)
        sy = min(max(round(sy), 0), self.sy - 1)
        return sx, sy

    def to_dst_coord(self, sx, sy):
        "变换前坐标->变换后坐标"
        dx, dy = np.matmul(self.m2x3_to_dst, np.array([sx, sy, 1]).T)
        dx = min(max(round(dx), 0), self.dx - 1)
        dy = min(max(round(dy), 0), self.dy - 1)
        return dx, dy
