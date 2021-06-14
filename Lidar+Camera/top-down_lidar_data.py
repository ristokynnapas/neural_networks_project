"""

from plyfile import PlyData, PlyElement
import numpy as np
import cv2

def file_read(file):
    
    #Reading LIDAR laser beams (angles and corresponding distance data)
    
    f = PlyData.read(file)
    #measures = f.readlines()
    x = f['vertex'].data['x']
    y = f['vertex'].data['y']
    #print(f)
    #print(x)
    x_coordinates = []
    y_coordinates = []
    
    #for measure in measures:
        #print(measure)
        #x_coordinates.append(float(measure[0]))
        #y_coordinates.append(float(measure[1]))
    #x_coordinates = np.arrayx_coordinates)
    #y_coordinates = np.array(y_coordinates)
    #return x_coordinates, y_coordinates
    return x, y

x, y = file_read("00029302.ply")
b = np.zeros([800,800,3], np.uint8)

#image = np.stack((x, y), axis=-1)

#print(image)

for i in range(len(x)):
    #print(x[i])
    b[int(x[i]), int(y[i])] = [0,0,255]

b[100, 100] = [255,255,255]   

#print(b.shape)
#b[-19,-5] = [0,0,255]
cv2.imshow("Color Image", b)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(file,
                          side_range=(-75, 75),
                          fwd_range=(-75,75),
                          res=0.1,
                          min_height = -100,
                          max_height = 100,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """

    f = PlyData.read(file)
    #measures = f.readlines()
    x_lidar = f['vertex'].data['x']
    y_lidar = f['vertex'].data['y']
    z_lidar = f['vertex'].data['z']

    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    # SAVE THE IMAGE
    if saveto is not None:
        im.save(saveto)
    else:
        im.show()

#birds_eye_point_cloud("00018889.ply")
birds_eye_point_cloud("00000146.ply")
