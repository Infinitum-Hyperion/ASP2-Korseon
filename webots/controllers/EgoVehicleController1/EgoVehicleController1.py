"""EgoVehicleController1 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import sys
import os

sys.path.append(os.path.abspath("/Users/OBSiDIAN/Downloads/Shelves/VSCode/Repositories/The Hyperion Project/markhor/markhor_sdk/python_3_9/communication/"))
from lightweight_communication_bridge import LCB

sys.path.append(os.path.abspath("/Applications/Webots.app/Contents/lib/controller/python39/"))
from vehicle import Driver
from controller import Lidar

def onMsg(payload: dict[str, object]) -> None:
    pass

lcb = LCB(onMsg)
driver = Driver()

class VehicleState:
    def __init__(self, timestep:int) -> None:
        self.sensor_lidar = driver.getDevice("lidarFront")
        self.sensor_lidar.enable(timestep)
        self.sensor_lidar.enablePointCloud()
        # https://cyberbotics.com/doc/reference/camera?tab-language=python#description
        self.sensor_camera = driver.getDevice("cameraFront")
        self.sensor_camera.enable(timestep)

    def getProcPointCloud(self):
        pointCloud = vstate.sensor_lidar.getPointCloud()
        pointCloudData: list[dict[str, object]] = []
        for point in pointCloud:
            pointCloudData.append({
                'x': point.x,
                'y': point.y,
                'z': point.z,
                'layer_id': point.layer_id,
                'time': point.time,
            })
        return pointCloudData
    
    def snapshot(self) -> dict[str, object]:
        return {
            'cameraFront': self.sensor_camera.getImage(), # bytes
            'lidarFront': self.getProcPointCloud(), # list[dict[str, object]]
        }


def simulatorLoop():
    # Duration of 1 timestep, in milliseconds
    timestep = int(driver.getBasicTimeStep())
    print(f"Step duration: {timestep}")
    timesteps: int = 0
    vstate = VehicleState(timestep)

    while driver.step() != -1:
        timesteps += 1

        print(f"Step #{timesteps}")
        if (timesteps == 200):
            lcb.close()
    pass

def main():
    simulatorLoop()

main()