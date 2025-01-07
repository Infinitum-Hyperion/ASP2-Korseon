import numpy as np
import cv2
from filter_canvas import FilterCanvas

class LaneBoundaryFilter:
  def __init__(self, id, a, b, c):
    self.id = id
    # Initialize state vector and covariance matrix
    self.x = np.array([a, b, c, 0, 0, 0])
    self.P = np.eye(6)    # Initial state covariance

    # Define system matrices
    self.dt = 0.1  # Time step
    self.A = np.array([[1, self.dt, 0, 0, 0, 0],
                  [0, 1, self.dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, self.dt, 0],
                  [0, 0, 0, 0, 1, self.dt],
                  [0, 0, 0, 0, 0, 1]])

    self.H = np.eye(6)[:5, :]  # Measurement matrix (maps state to measurements)
    self.Q = np.eye(6) * 1e-2  # Process noise covariance
    self.R = np.eye(5) * 1e-1  # Measurement noise covariance

  def predict(self, linspaces, filter_canvas:FilterCanvas):
    # Predict step
    self.x = self.A @ self.x
    self.P = self.A @ self.P @ self.A.T + self.Q
    filter_canvas.addPredicted(self.visualise(self.x, linspaces))
    return self.x

  def update(self, a, b, c, curvature, heading, linspaces, filter_canvas:FilterCanvas):
    # Update step
    z = np.array([a, b, c, curvature, heading])
    y = z - self.H @ self.x                                # Measurement residual
    S = self.H @ self.P @ self.H.T + self.R                # Residual covariance
    K = self.P @ self.H.T @ np.linalg.inv(S)               # Kalman gain
    self.x = self.x + K @ y                                # Updated state estimate
    self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P   # Updated covariance estimate
    filter_canvas.addUpdated(self.visualise(self.x, linspaces))
    return self.x

  def visualise(self, x, linspaces):
    if len(linspaces)== 0: return
    a, b, c = x[:3]
    # Generate points along the fitted polynomial
    x_vals = np.linspace(linspaces[self.id][0], linspaces[self.id][1], 100)
    y_vals = (a * x_vals**2 + b * x_vals + c).astype(int)
    points = [(x_val, y_val) for x_val, y_val in zip(x_vals, y_vals)]
    return points