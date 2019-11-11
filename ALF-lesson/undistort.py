import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

points= []
# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    #gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray = dst
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found: 
            # a) draw corners
    gray = cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
    print("{}".format(corners.shape))
    print("Using corners: " + str(corners[0:3]))
    p1 = corners[0,0,:]
    p2 = corners[7,0,:]
    p4 = corners[47, 0, :]
    p3 = corners[40, 0, :]
    points = [p1, p2, p3, p4]
    src = np.float32(corners[0:3])
    min_x = np.min(corners[:,:,0])
    max_x = np.max(corners[:, :, 0])
    min_y = np.min(corners[:, :, 1])
    max_y = np.max(corners[:, :, 1])
    print(str([min_x, min_y, max_x, max_y]))
    print(str(p1) + " " + str(p2) + " " + str(p3) + " " + str(p4))
    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([p1, [p4[0], p1[1]], [p1[0], p4[1]], p4])
    dst = np.float32([[75, 75], [1200, 75], [75, 875], [1200, 875]])
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    M = cv2.getPerspectiveTransform(src, dst)
#    warped = np.copy(gray) 
    img_size = gray.shape[1::-1]
    warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

