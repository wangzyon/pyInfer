import cv2


class DetectionPlot():
    label_colors = (
        (255, 0, 255),
        (0, 0, 255),
    )
    corner_colors = (
        (0, 255, 0),
        (0, 255, 0),
    )

    def __init__(self, image, bboxes) -> None:
        self.image = image
        self.bboxes = bboxes

    def plot(self):
        for bbox in self.bboxes:
            left, top, right, bottom, label = round(bbox.left), round(bbox.top), round(bbox.right), round(
                bbox.bottom), int(bbox.label)
            self.plot_rectangle(left, top, right, bottom, label)
            self.plot_label_type(left, top, label, bbox.labelname)
            self.plot_box_corner(left, top, right, bottom, label)
        return self

    def plot_rectangle(self, left, top, right, bottom, label, thickness=2):
        cv2.rectangle(self.image, (left, top), (right, bottom), color=self.label_colors[label], thickness=thickness)

    def plot_label_type(self, left, top, label, labelname):
        labelSize = cv2.getTextSize(labelname + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        if top - labelSize[1] - 3 < 0:
            cv2.rectangle(
                self.image, (left, top + 2), (left + labelSize[0], top + labelSize[1] + 3),
                color=self.label_colors[label],
                thickness=-1)
            cv2.putText(
                self.image,
                labelname, (left, top + labelSize[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0),
                thickness=1)
        else:
            cv2.rectangle(
                self.image, (left, top - labelSize[1] - 3), (left + labelSize[0], top - 3),
                color=self.label_colors[label],
                thickness=-1)
            cv2.putText(self.image, labelname, (left, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

    def plot_box_corner(self, left, top, right, bottom, label, length=10, thickness=3):
        # Top Left

        cv2.line(self.image, (left, top), (left + length, top), self.corner_colors[label], thickness=thickness)
        cv2.line(self.image, (left, top), (left, top + length), self.corner_colors[label], thickness=thickness)
        # Top Right
        cv2.line(self.image, (right, top), (right - length, top), self.corner_colors[label], thickness=thickness)
        cv2.line(self.image, (right, top), (right, top + length), self.corner_colors[label], thickness=thickness)
        # Bottom Left
        cv2.line(self.image, (left, bottom), (left + length, bottom), self.corner_colors[label], thickness=thickness)
        cv2.line(self.image, (left, bottom), (left, bottom - length), self.corner_colors[label], thickness=thickness)
        # Bottom Right
        cv2.line(self.image, (right, bottom), (right - length, bottom), self.corner_colors[label], thickness=thickness)
        cv2.line(self.image, (right, bottom), (right, bottom - length), self.corner_colors[label], thickness=thickness)

    def save(self, path):
        cv2.imwrite(path, self.image)