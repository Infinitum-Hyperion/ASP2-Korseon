import time
import math
import random
import string

from bisenet.model import BiSeNetModel
import postproc
from filter_canvas import FilterCanvas
from lane_boundary_filter import LaneBoundaryFilter

filters: dict[str, LaneBoundaryFilter] = {}
bisenet_model = BiSeNetModel()
filter_colors = {}
prevLinspaces = {}

def generateId(): return ''.join(random.choices(string.ascii_letters + string.digits, k=5))

def euclideanDist(a1, b1, c1, a2, b2, c2):
  return math.sqrt((a1-a2)**2+(b1-b2)**2+(c1-c2)**2)

def pipeline(timestepNum):
  imgPath = f'./images/timestep_{timestepNum}.png'
  filters_canvas: FilterCanvas = FilterCanvas(imgPath)
  # Generate filter predictions
  if (len(filters) != 0):
    predictions = {idx: filter.predict(prevLinspaces, filters_canvas) for idx, filter in filters.items()}
    print(f"Predictions: {predictions}")
  # Run lane detection
  lane_mask = bisenet_model.predict(imgPath, 0.71)
  spatial_mask = postproc.spatialMask(lane_mask)
  lines_mask = postproc.linesMask(spatial_mask)
  contours = postproc.findContours(lines_mask, lane_mask)
  filtered_labels_by_area, cluster_colors, filtered_contours, canvas = postproc.runDBSCAN(contours, lines_mask, imgPath)

  # Fit polynomials
  coeffsMap, linspacesMap = postproc.fitPolynomials(filtered_labels_by_area, cluster_colors, filtered_contours, imgPath)
  print(f"CoeffsMap: {coeffsMap}")

  # Update Kalman filters
  refined_coeffs = {}
  if len(filters) != 0:
    for filter_id, prediction in predictions.items():
      maxScore = -1
      bestMsmt = None
      for _, msmt in coeffsMap.items():
        a_pred, b_pred, c_pred = prediction[:3]
        score = euclideanDist(a_pred, b_pred, c_pred, msmt[0], msmt[1], msmt[2])
        if score > maxScore:
          maxScore = score
          bestMsmt = msmt
      print(f"Best msmt: {bestMsmt} for {prediction} [{filter_id}]")
      refined_coeffs[filter_id] = filters[filter_id].update(bestMsmt[0], bestMsmt[1], bestMsmt[2], 0, 0, prevLinspaces, filters_canvas)

  else:
    for id, coeff in coeffsMap.items():
      filter_id = generateId()
      filters[filter_id] = LaneBoundaryFilter(id, coeff[0], coeff[1], coeff[2])
      refined_coeffs[filter_id] = coeff[0], coeff[1], coeff[2]
  filters_canvas.renderAll()
  prevLinspaces.clear()
  prevLinspaces.update(linspacesMap)
  return refined_coeffs

def main():
  timesteps = ['40', '80', '120', '160', '200', '240', '280', '320']
  for timestep in timesteps:
    print(f"REFINED: {pipeline(timestep)}")
    cmd = input(">>")
    if cmd == 'e': exit(0)
    else:
      while cmd == 'r':
        print(f"REFINED: {pipeline(timestep)}")
        cmd = input(">>")

if __name__=='__main__': main()