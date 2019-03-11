import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

objpoints = []
imgpoints = []
nx = 9
ny = 6

images = glob.glob('camera_cal\\calibration*.jpg')
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for fname in images:

    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def create_threshold_binary_image(undist_img):

    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    gray_undist = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray_undist, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    min_th = 20
    max_th = 100

    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= min_th) & (scaled_sobel <= max_th)] = 1

    min_th_s = 170
    max_th_s = 255
    s_ch_binary = np.zeros_like(s_channel)
    s_ch_binary[(s_channel >= min_th_s) & (s_channel <= max_th_s)] = 1

    #color_binary = np.dstack((np.zeros_like(sobel_x_binary), sobel_x_binary, s_ch_binary)) * 255
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[(s_ch_binary == 1) | (sobel_x_binary == 1)] = 1
    #plt.imshow(combined_binary, cmap='gray')
    #plt.show()

test_img = mpimg.imread('test_images\\straight_lines1.jpg')

undist = undistort_image(test_img, mtx, dist)
create_threshold_binary_image(undist)
