"""EgoVehicleController1 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import sys
import os
import math
import base64
from io import BytesIO
from PIL import Image
import warnings

sys.path.append(os.path.abspath("/Users/OBSiDIAN/Downloads/Shelves/VSCode/Repositories/The Hyperion Project/markhor/markhor_sdk/python_3_9/telemetry"))
from lightweight_communication_bridge import LCB

sys.path.append(os.path.abspath("/Applications/Webots.app/Contents/lib/controller/python39/"))
from vehicle import Driver
from controller import Lidar

def onMsg(payload: dict[str, object]) -> None:
    pass

# lcb = LCB(onMsg, '0.0.0.0', '8080')
driver = Driver()

class VehicleState:
    def __init__(self, timestep:int) -> None:
        self.sensor_lidar = driver.getDevice("lidarFront")
        self.sensor_lidar.enable(timestep)
        self.sensor_lidar.enablePointCloud()
        # https://cyberbotics.com/doc/reference/camera?tab-language=python#description
        self.sensor_camera_top = driver.getDevice("cameraTop")
        self.sensor_camera_top.enable(timestep)
        self.sensor_gps = driver.getDevice("gps")
        self.sensor_gps.enable(timestep)

    def getProcPointCloud(self):
        pointCloud = self.sensor_lidar.getPointCloud()
        pointCloudData: list[dict[str, object]] = []
        for point in pointCloud:
            if point.x != math.inf and point.x !=-math.inf and point.y != math.inf and point.y !=-math.inf and point.z != math.inf and point.z !=-math.inf:
                pointCloudData.append({
                    'x': point.x,
                    'y': point.y,
                    'z': point.z,
                    'layer_id': point.layer_id,
                    'time': point.time,
                })
        return pointCloudData

    def getImg(self):
        img  = Image.frombytes("RGBA", (self.sensor_camera_top.getWidth(), self.sensor_camera_top.getHeight()), self.sensor_camera_top.getImage())
        return img

    def getProcImgBytes(self, img: Image.Image):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def getGpsVector(self):
        return self.sensor_gps.getValues()

    def getDash(self) -> dict[str, object]:
        return {
            'rpm': driver.getRpm(),
            'speed': driver.getCurrentSpeed(),
            'gear': driver.getGearNumber(),
            'steer': driver.getSteeringAngle(),
        }
    
    def snapshot(self) -> dict[str, object]:
        return {
            'msgType': 'snapshot',
            # 'cameraFront': self.getProcImgBytes(self.getImg()), # str(b64(byte stream))
            'cameraTop': self.getProcImgBytes(self.getImg()),
            'lidarFront': self.getProcPointCloud(), # list[dict[str, object]]
            'gps': self.getGpsVector(),
            'dash': self.getDash(),
        }

    def navUpdate(self) -> dict[str, object]:
        return {
            'msgType': 'nav-update',
            'gps': self.getGpsVector(),
            'dash': self.getDash(),
        }


def simulatorLoop():
    # Duration of 1 timestep, in milliseconds
    timestep = int(driver.getBasicTimeStep())
    print(f"Step duration: {timestep}")
    timesteps: int = 0
    vstate = VehicleState(timestep)
    driver.setSteeringAngle(0)
    driver.setCruisingSpeed(20)
    data = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        while driver.step() != -1:
            timesteps += 1

            print(f"Step #{timesteps}")
            """ if (timesteps == 80):
                lcb.send('asp2-korseon.korseon-main', {'source': 'asp2-korseon.vehicle-controller', **vstate.snapshot()})
            if (timesteps % 50 == 0):
                lcb.send('asp2-korseon.korseon-main', {'source': 'asp2-korseon.vehicle-controller', **vstate.navUpdate()})
             """
            if (timesteps % 40 == 0):
                # img = vstate.getImg()
                # img.save(f'timestep_{timesteps}.png')
                data[timesteps] = vstate.getDash()
            if (timesteps == 160):
                driver.setSteeringAngle(-0.01)
            if (timesteps == 280):
                driver.setSteeringAngle(0)
            if (timesteps == 500):
                print(data)
            
    pass

def main():
    simulatorLoop()

main()