import cv2
import numpy as np
class FilterCanvas:
  def __init__(self, imgPath: str):
    self.original_image = cv2.imread(imgPath)
    self.id = f'fltcnvs_{imgPath.split("_")[-1]}.jpg'

  def addLine(self, points, color):
    height, width, _ = self.original_image.shape
    points = np.array([point for point in points if 0 <= point[1] < height], np.int32)
    cv2.polylines(self.original_image, [points], color=color, isClosed=False, thickness=2)
  
  def addPredicted(self, points): self.addLine(points, color=(255, 0, 0))
  
  def addUpdated(self, points): self.addLine(points, color=(0, 255, 0))
  
  def renderAll(self):
    cv2.imwrite(self.id, self.original_image)