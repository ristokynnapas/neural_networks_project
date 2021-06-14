# This script is for generating video files from image files using opencv3

import glob
import cv2 as cv
import collections

image_array = collections.deque([])
for image_file in glob.glob("C:/Users/Personal/Desktop/WindowsNoEditor/PythonAPI/examples/_out/*.png"):
    image = cv.imread(image_file)
    image_array.append(image)

# the size of the images should be the same (last image size) ->
height, width, layers = image.shape
size = (width,height)

# video codec and params ->
result = cv.VideoWriter("Result videos/Proof_of_concept1.avi", cv.VideoWriter_fourcc(*"MJPG"), 10, size)

# generating frame-by-frame the video file in result variable -> 
for frame in range(len(image_array)):
    result.write(image_array[frame])

# when done, exit safely ->
cv.destroyAllWindows()
result.release()

