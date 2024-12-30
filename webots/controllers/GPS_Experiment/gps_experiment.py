"""EgoVehicle controller for GPS Experiment."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import sys
import os
import math
import base64
from io import BytesIO
from PIL import Image

sys.path.append(os.path.abspath("/Users/OBSiDIAN/Downloads/Shelves/VSCode/Repositories/The Hyperion Project/markhor/markhor_sdk/python_3_9/telemetry"))
from lightweight_communication_bridge import LCB

sys.path.append(os.path.abspath("/Applications/Webots.app/Contents/lib/controller/python39/"))
from vehicle import Driver
from controller import Lidar

def onMsg(payload: dict[str, object]) -> None:
    pass

lcb = LCB(onMsg, '0.0.0.0', '8080')
driver = Driver()

class VehicleState:
    def __init__(self, timestep:int) -> None:
        self.sensor_gps = driver.getDevice("gps")
        self.sensor_gps.enable(timestep)

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
            'msgType': 'gps',
            'gps': self.getGpsVector(),
            'dash': self.getDash(),
        }


def simulatorLoop():
    # Duration of 1 timestep, in milliseconds
    timestep = int(driver.getBasicTimeStep())
    print(f"Step duration: {timestep}")
    timesteps: int = 0
    vstate = VehicleState(timestep)
    res = []
    while driver.step() != -1:
        timesteps += 1

        print(f"Step #{timesteps}")
        if (timesteps == 1):
            res.append(vstate.getGpsVector())
        if (timesteps % 2 == 0):
            lcb.send('asp2-korseon.korseon-main', {'source': 'asp2-korseon.vehicle-controller', **vstate.snapshot()})
        if (timesteps % 20 == 0):
            res.append(vstate.getGpsVector())
        if (timesteps == 560):
            res.append(vstate.getGpsVector())
        """             driver.setThrottle(0)
                    driver.setBrakeIntensity(0.3)
                    driver.setSteeringAngle(0.08)
                if (timesteps == 310):
                    driver.setBrakeIntensity(0)
                    driver.setThrottle(0.8)
                if (timesteps == 370):
                    driver.setSteeringAngle(0.11)
                    driver.setThrottle(0.99)
                if (timesteps == 460):
                    driver.setSteeringAngle(0)
                    driver.setThrottle(0.8) """
        if (timesteps == 500):
            driver.setThrottle(0)
            driver.setBrakeIntensity(0.8)
            driver.setSteeringAngle(0.1)
            print(res)
        if (timesteps == 570):
            driver.setBrakeIntensity(0)
            driver.setSteeringAngle(0)
            driver.setThrottle(1)
        if (timesteps == 1200):
            lcb.close()
    pass

def main():
    driver.setGear(1)
    driver.setThrottle(0.8)
    simulatorLoop()

main()