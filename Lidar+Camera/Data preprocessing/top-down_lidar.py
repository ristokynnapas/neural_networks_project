"""
This file is for creating a top-down view of the lidar point cloud data.

The code is a modified and merged version of pieces found in th WWW.

"""
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np

# scale the image data ->
def rescale(a, min, max):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

# generate top-down view
def top_down_view(file, widht_range=(-75, 75), fwd_range=(-75,75), resolution=0.1, min_height = -100,
                        max_height = 100):

    # get the coordinates from the file ->
    f = PlyData.read(file)
    x_coordinate = f['vertex'].data['x']
    y_coordinate = f['vertex'].data['y']
    z_coordinate = f['vertex'].data['z']

    # select the data from a rectangular area ->
    length = np.logical_and((x_coordinate > length_range[0]), (x_coordinate < length_range[1]))
    width = np.logical_and((y_coordinate > -widht_range[1]), (y_coordinate < -widht_range[0]))
    area = np.argwhere(np.logical_and(length,width)).flatten()

    # convert data to pixel type
    x_direction = (-y_coordinate[area]/resolution).astype(np.int32) 
    y_direction = (x_coordinate[area]/resolution).astype(np.int32)  

    x_direction -= int(np.floor(widht_range[0]/resolution))
    y_direction -= int(np.floor(length_range[0]/resolution))

    # select appropriate height values ->
    height = np.clip(z_direction = z_coordinate[area], z_min=min_height, z_max=max_height)
    height  = rescale(height, min=min_height, max=max_height)

    # write pixel values ->
    x_max = int((widht_range[1] - widht_range[0])/resolution)
    y_max = int((length_range[1] - length_range[0])/resolution)
    image = np.zeros([y_max, x_max], dtype=np.uint8)
    image[-y_direction, x_direction] = height # -y because images start from top left

    image = Image.fromarray(image)
    image.show()

#top_down_view("00018889.ply")
top_down_view("00000197.ply")

