from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import sys
import random
import time
import cv2
import numpy as np
import h5py
import keras
from keras.models import load_model
try:
    sys.path.append(glob.glob(r'C:\Users\fried\OneDrive\Desktop\Masters\Spring 2020-2021\Neural network\Project\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
IM_WIDTH=1280
IM_HEIGHT=720
i=0
# modelsteer=load_model('steering3.h5')
# modelthrottle=load_model('throttle.h5')
def process_img(image):
    global i
    raw=np.array(image.raw_data)
    img=raw.reshape((IM_HEIGHT,IM_WIDTH,4))
    finalimg = img[:,:,:3]
    cv2.imshow("s",finalimg)
    if i<10:
        cv2.imwrite(f'0{i}.png',finalimg)
    else:
        cv2.imwrite(f'{i}.png',finalimg)
    i+=1
    # finalimage=cv2.resize(finalimg, (215, 120))
    # finalimage=np.reshape(finalimage,(1,finalimage.shape[0], finalimage.shape[1], finalimage.shape[2]))
    # steering=float(modelsteer.predict(finalimage/255)[0][0])
    # throttled=float(modelthrottle.predict(finalimage/255)[0][0])
    # vehicle.apply_control(carla.VehicleControl(throttle=throttled,steer=steering))
    cv2.waitKey(1)
    return finalimg
def process_img2(image):
    raw=np.array(image.raw_data)
    img=raw.reshape((IM_HEIGHT,IM_WIDTH,4))
    finalimg = img[:,:,:3]
    cv2.imshow("z",finalimg)
    cv2.waitKey(1)
    return finalimg
def process_img3(image):
    raw=np.array(image.raw_data)
    img=raw.reshape((IM_HEIGHT,IM_WIDTH,4))
    finalimg = img[:,:,:3]
    cv2.imshow("d",finalimg)
    cv2.waitKey(1)
    return finalimg

actor_list=[]
try:
    client=carla.Client("localhost",2000)
    client.set_timeout(10.0)

    world=client.get_world()
    blueprint_library=world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]
    # spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_point=world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    # spawn_point2 = carla.Transform(carla.Location(x=2.5,z=0.7,y=0.5))
    # spawn_point3 = carla.Transform(carla.Location(x=2.5,z=0.7,y=-0.5))
    sensor_center=world.spawn_actor(cam_bp,spawn_point, attach_to=vehicle)
    # sensor_left=world.spawn_actor(cam_bp,spawn_point2, attach_to=vehicle)
    # sensor_right=world.spawn_actor(cam_bp,spawn_point3, attach_to=vehicle)
    actor_list.append(sensor_center)
    # actor_list.append(sensor_left)
    # actor_list.append(sensor_right)
    sensor_center.listen(lambda data: process_img(data))
    # sensor_left.listen(lambda data: process_img2(data))
    # sensor_right.listen(lambda data: process_img3(data))
    time.sleep(1000)
finally:
    for actor in actor_list:
        actor.destroy()
    print('Clean!')