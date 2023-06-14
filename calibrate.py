# Import required modules
import cv2
import numpy as np
import os
import glob
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import cv2

'''
rgb_folder = './calibrate_images/'
rgb_file = 'rosbag2_2023_04_20-16_35_55'

# create reader instance and open for reading
with Reader(rgb_folder+rgb_file) as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/color/preview/image':
            img_msg = deserialize_cdr(rawdata, connection.msgtype)
#             print(img_msg.header.frame_id)

rgb = img_msg.data.reshape(250,250,3)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(rgb, (640,480), interpolation = cv2.INTER_AREA)
plt.imshow(gray)
plt.show()

from PIL import Image
gray = Image.fromarray(gray, "RGB")
gray.save("./calibrate_images/sample4.png")

'''
# Define the dimensions of checkerboard
CHECKERBOARD = (6, 8)
  
  
# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  

# Vector for 3D points
threedpoints = []
  
# Vector for 2D points
twodpoints = []
  
  
#  3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] 
                      * CHECKERBOARD[1], 
                      3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
  
  
# Extracting path of individual image stored
folder = './calibrate_images/'
for filename in os.listdir(folder):
    if filename.endswith('png'):
        image = cv2.imread(os.path.join(folder,filename))
        image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_LINEAR)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)
    
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
    
            twodpoints.append(corners2)
    
            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, 
                                            CHECKERBOARD, 
                                            corners2, ret)
    
        cv2.imshow('img', image)
        cv2.waitKey(0)
  
cv2.destroyAllWindows()
  
h, w = image.shape[:2]
  
  
# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)
  
  
# Displaying required output
print(" Camera matrix:")
print(matrix)
  
print("\n Distortion coefficient:")
print(distortion)
  
print("\n Rotation Vectors:")
print(r_vecs)
  
print("\n Translation Vectors:")
print(t_vecs)


'''
Camera matrix:
[[380.35137528   0.         124.56690371]
 [  0.         382.9813493  143.1272969 ]
 [  0.           0.           1.        ]]

 Distortion coefficient:
[[ 3.78705295e-01 -2.88260992e+00  2.57854982e-02 -2.27272377e-03
   1.62991977e+01]]

 Rotation Vectors:
(array([[-0.00963123],
       [-0.00907175],
       [-1.56096746]]), array([[-0.03237968],
       [ 0.01172053],
       [-1.38039352]]), array([[0.00830266],
       [0.03486729],
       [1.53880035]]), array([[ 0.36317318],
       [-0.37621531],
       [ 1.53220419]]))

 Translation Vectors:
(array([[-3.84023461],
       [ 0.91055252],
       [18.58111833]]), array([[-4.43284102],
       [-0.34163618],
       [20.72506611]]), array([[ 3.67272497],
       [-5.15639586],
       [20.52184813]]), array([[ 3.35116942],
       [-3.38308484],
       [16.89381073]]))
'''