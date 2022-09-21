from pydantic import BaseModel
import numpy as np
from typing import Union


class QuadrangleBBox(BaseModel):

    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    x3: float = 0
    y3: float = 0
    x4: float = 0
    y4: float = 0
    area: float = 0
    confidence: float = 0
    label: int = 0
    labelname: str = ""
    keepflag: bool = True

    @property
    def coords(self):
        """坐标值"""
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]

    def set_coords(self, coords):
        """坐标值"""
        assert len(coords) == 8
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = coords
        return self

    @property
    def xcoords(self):
        """坐标值"""
        return [self.x1, self.x2, self.x3, self.x4]

    def set_xcoords(self, xcoords):
        assert len(xcoords) == 4
        self.x1, self.x2, self.x3, self.x4 = xcoords
        return self

    @property
    def ycoords(self):
        """坐标值"""
        return [self.y1, self.y2, self.y3, self.y4]

    def set_ycoords(self, ycoords):
        assert len(ycoords) == 4
        self.y1, self.y2, self.y3, self.y4 = ycoords
        return self

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def left(self):
        return min(self.x1, self.x2, self.x3, self.x4)

    @property
    def right(self):
        return max(self.x1, self.x2, self.x3, self.x4)

    @property
    def top(self):
        return min(self.y1, self.y2, self.y3, self.y4)

    @property
    def bottom(self):
        return max(self.y1, self.y2, self.y3, self.y4)

    def reset_origin(self, ox, oy):
        """更新原点"""
        self.xcoords = [self.x1 - ox, self.x2 - ox, self.x3 - ox, self.x4 - ox]
        self.ycoords = [self.y1 - oy, self.y2 - oy, self.y3 - oy, self.y4 - oy]
        return self

    def clip(self, xmin, xmax, ymin, ymax):
        self.xcoords = list(map(lambda x: min(max(x, xmin), xmax), self.xcoords))
        self.ycoords = list(map(lambda y: min(max(y, ymin), ymax), self.ycoords))
        return self
    
    
    @property
    def level(self):
        return len(set(self.ycoords)) == len(set(self.xcoords)) == 2

    def __mul__(self, scalar: Union[int, float]):
        """坐标乘"""
        self.coords = list(np.array(self.coords) * scalar)
        return self

    def __setattr__(self, key, val):
        # @coords.setter is not support in BaseModel, thus, modify __setatter__
        method = self.__config__.property_set_methods.get(key)
        if method is None:
            super().__setattr__(key, val)
        else:
            getattr(self, method)(val)

    class Config:
        property_set_methods = {"coords": "set_coords", "xcoords": "set_xcoords", "ycoords": "set_ycoords"}
