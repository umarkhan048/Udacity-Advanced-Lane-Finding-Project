import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

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
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
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
    return combined_binary, h_channel, l_channel

def perspective_transform(cmb_bin_img):

    src = np.float32(
        [[580, 460],
         [1110, 720],
         [205, 720],
         [703, 460]])

    dst = np.float32(
        [[320, 0],
         [960, 720],
         [320, 720],
         [960, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    image_size = (cmb_bin_img.shape[1], cmb_bin_img.shape[0])
    warped_img = cv2.warpPerspective(cmb_bin_img, M, image_size, flags=cv2.INTER_LINEAR)
    return warped_img, Minv

def find_lane_pixels(img_warped):
    bottom_half = img_warped[img_warped.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((img_warped, img_warped, img_warped))*255
    mid_point = np.int(histogram.shape[0]//2)
    left_x_base = np.argmax(histogram[:mid_point])
    right_x_base = np.argmax(histogram[mid_point:]) + mid_point

    nwindows = 15
    margin = 70
    minpix = 50

    win_height = np.int(img_warped.shape[0]//nwindows)
    non_zero_pixels = img_warped.nonzero()
    non_zero_y = np.array(non_zero_pixels[0])
    non_zero_x = np.array(non_zero_pixels[1])

    left_x_current = left_x_base
    right_x_current = right_x_base

    left_lane_pixels = []
    right_lane_pixels = []

    for window in range(nwindows):
        win_y_low = img_warped.shape[0] - (window + 1) * win_height
        win_y_high = img_warped.shape[0] - window * win_height
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),(win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (non_zero_x >= win_xleft_low) & (non_zero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (non_zero_x >= win_xright_low) & (non_zero_x < win_xright_high)).nonzero()[0]

        left_lane_pixels.append(good_left_inds)
        right_lane_pixels.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(non_zero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = np.int(np.mean(non_zero_x[good_right_inds]))

    left_lane_pixels = np.concatenate(left_lane_pixels)
    right_lane_pixels = np.concatenate(right_lane_pixels)

    left_x = non_zero_x[left_lane_pixels]
    left_y = non_zero_y[left_lane_pixels]
    right_x = non_zero_x[right_lane_pixels]
    right_y = non_zero_y[right_lane_pixels]

    return left_x, left_y, right_x, right_y, out_img

def fit_polynomial(left_x, left_y, right_x, right_y, transformed_img):
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    ploty = np.linspace(0, transformed_img.shape[0] - 1, transformed_img.shape[0])

    left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit_x, right_fit_x, ploty

def measure_curvature(ploty, left_x, right_x):

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    left_fit = np.polyfit(ploty*ym_per_pix, left_x*xm_per_pix, 2)
    right_fit = np.polyfit(ploty*ym_per_pix, right_x*xm_per_pix, 2)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad

def draw_on_image(warped_img, left_fitx, right_fitx, ploty, Minv, undist, image):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def process_image(image):
    undist = undistort_image(test_img, mtx, dist)
    combined_binary_img, h_channel, l_channel = create_threshold_binary_image(undist)
    transformed_img, Minv = perspective_transform(combined_binary_img)
    left_x, left_y, right_x, right_y, out_img = find_lane_pixels(transformed_img)
    left_fit_x, right_fit_x, ploty = fit_polynomial(left_x, left_y, right_x, right_y, transformed_img)
    left_curve_radius, right_curve_radius = measure_curvature(ploty, left_fit_x, right_fit_x)
    result = draw_on_image(transformed_img, left_fit_x, right_fit_x, ploty, Minv, undist, test_img)
    print(left_curve_radius)
    print(right_curve_radius)
    plt.imshow(result)
    plt.show()


test_img = mpimg.imread('test_images\\straight_lines1.jpg')
#test_img = mpimg.imread('test_images\\straight_lines2.jpg')
#test_img = mpimg.imread('test_images\\test1.jpg')
#test_img = mpimg.imread('test_images\\test2.jpg')
#test_img = mpimg.imread('test_images\\test3.jpg')
#test_img = mpimg.imread('test_images\\test4.jpg')
#test_img = mpimg.imread('test_images\\test5.jpg')
#test_img = mpimg.imread('test_images\\test6.jpg')

process_image(test_img)
# out_img[left_y, left_x] = [255, 0, 0]
# out_img[right_y, right_x] = [0, 0, 255]
# plt.imshow(out_img)
# plt.plot(left_fit_x, ploty, color='yellow')
# plt.plot(right_fit_x, ploty, color='yellow')
# plt.show()

# plt.subplot(2,3,1)
# plt.imshow(test_img)
# plt.subplot(2,3,2)
# plt.imshow(transformed_img)
# plt.subplot(2,3,3)
# plt.plot(img_histogram)
# plt.subplot(2,3,4)
# plt.imshow(combined_binary_img)
# plt.subplot(2,3,5)
# plt.imshow(h_channel)
# plt.subplot(2,3,6)
# plt.imshow(l_channel)
# plt.show()