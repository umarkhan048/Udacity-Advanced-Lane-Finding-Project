import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip


left_fit_poly = []
right_fit_poly = []

objpoints = []  # 3D points
imgpoints = []  # 2D points

# nx and ny define the number of corners in x and y direction
nx = 9
ny = 6

# Using glob function, all the calibration images are used to calculate parameters
images = glob.glob('camera_cal\\calibration*.jpg')
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for fname in images:

    # for each chessboard image, gray scale conversion is done
    # the opencv function findChessboardCorners is used to detect corners
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

        # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        a=1

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort_image(img_in, mtx_in, dist_in):

    # This is the function which undistorts using the calibration parameters
    dst = cv2.undistort(img_in, mtx_in, dist_in, None, mtx_in)
    return dst


def create_threshold_binary_image(undist_img):

    # RGB to HLS Conversion
    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)

    # using s channel and applying thresholds for lane detection
    s_channel = hls[:, :, 2]
    thresh_s_ch = (100, 255)    # Threshold values found by testing
    s_ch_binary = np.zeros_like(s_channel)
    s_ch_binary[(s_channel >= thresh_s_ch[0]) & (s_channel <= thresh_s_ch[1])] = 1

    # Sobel operator is applied to the undistorted gray image in x direction
    # The sudden changes in image pixels are detected
    gray_undist = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    sobel_kernel = 5
    sobelx = cv2.Sobel(gray_undist, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    thresh_sobel = (20, 100)    # Threshold values are found by testing
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= thresh_sobel[0]) & (scaled_sobel <= thresh_sobel[1])] = 1

    # After some testing, it was found that in some specific situations, the r-channel
    # does a better job of detecting right lanes than s channel, specially when light
    # colored road patches are in the frame.
    r_channel = undist_img[:, :, 0]
    thresh_r = (220, 255)   # Threshold values are found by testing
    binary_r = np.zeros_like(r_channel)
    binary_r[(r_channel > thresh_r[0]) & (r_channel <= thresh_r[1])] = 1

    # Creating a combined image with sobel, binary red and s channel binary
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[(sobel_x_binary == 1) | (binary_r == 1) | (s_ch_binary == 1)] = 1
    return combined_binary


def perspective_transform(cmb_bin_img):

    # trapezoid to be transformed
    src = np.float32(
        [[580, 460],
         [1110, 720],
         [205, 720],
         [703, 460]])

    # rectangular area on which the above trapezoid is warped to
    dst = np.float32(
        [[320, 0],
         [960, 720],
         [320, 720],
         [960, 0]])

    # Getting perspective transform matrix and warping the image
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    image_size = (cmb_bin_img.shape[1], cmb_bin_img.shape[0])
    warped_img = cv2.warpPerspective(cmb_bin_img, M, image_size, flags=cv2.INTER_LINEAR)
    return warped_img, Minv


def find_lane_pixels(img_warped):

    # The histogram of the lower bottom of the image is taken.
    # High values in the histogram point towards lanes in those regions
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img_warped[img_warped.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((img_warped, img_warped, img_warped))*255
    mid_point = np.int(histogram.shape[0]//2)
    left_x_base = np.argmax(histogram[:mid_point])
    right_x_base = np.argmax(histogram[mid_point:]) + mid_point

    # Sliding windows method is used to find lanes
    nwindows = 9   # Number of sliding windows
    margin = 80     # Width of the window
    minpix = 50     # Minimum number of pixels to be found to recenter window

    # Size of each window is image height divided by number of windows
    win_height = np.int(img_warped.shape[0]//nwindows)
    non_zero_pixels = img_warped.nonzero()
    non_zero_y = np.array(non_zero_pixels[0])
    non_zero_x = np.array(non_zero_pixels[1])

    # Current position to be updated later for each window
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Empty lists to receive the indices of left and right lane pixels
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Upper and lower boundaries of the window
        win_y_low = img_warped.shape[0] - (window + 1) * win_height
        win_y_high = img_warped.shape[0] - window * win_height

        # left and right boundaries of the window
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        # Draw the windows on te visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),(win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identifying non zero pixel  in x and y within the window
        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (non_zero_x >= win_xleft_low) & (non_zero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (non_zero_x >= win_xright_low) & (non_zero_x < win_xright_high)).nonzero()[0]

        # Appending the indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter the upcoming window if minpix number of pixels are found
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(non_zero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = np.int(np.mean(non_zero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = non_zero_x[left_lane_inds]
    left_y = non_zero_y[left_lane_inds]
    right_x = non_zero_x[right_lane_inds]
    right_y = non_zero_y[right_lane_inds]

    return left_x, left_y, right_x, right_y, out_img


def fit_polynomial(transformed_img):

    # the pixel where non zero values are found are returned
    left_x, left_y, right_x, right_y, out_img = find_lane_pixels(transformed_img)

    # polynomial coefficients for left and right lanes are found and lines are created
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    ploty = np.linspace(0, transformed_img.shape[0] - 1, transformed_img.shape[0])

    # Using the polynomial coefficients found from above, the left and right lane
    # points are found
    left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Update previous poly - in this case they will be updated from zero
    global left_fit_poly
    left_fit_poly = left_fit
    global right_fit_poly
    right_fit_poly = right_fit

    # Visualization ##
    # Colors in the left and right lane regions
    out_img[left_y, left_x] = [255, 0, 0]
    out_img[right_y, right_x] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fit_x, ploty, color='yellow')
    # plt.plot(right_fit_x, ploty, color='yellow')

    return out_img, left_fit_x, right_fit_x, ploty


def search_around_polynomial(binary_warped, left_fit_prev, right_fit_prev):

    # This the margin around previously found polynomial where lanes will be searched
    margin = 100

    # Grab the activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Here the area of search based on activated x-values within the +/- margin
    # of the previously found polynomial function is set
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy +
                    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) +
                    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy +
                    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) +
                    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Extract left and right lane activated pixels
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Using polynomial coefficients from above, left and right lane points are found
    left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Update array for previous polynomials
    global left_fit_poly
    left_fit_poly = left_fit
    global right_fit_poly
    right_fit_poly = right_fit

    # Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')
    # End visualization steps ##

    return result, left_fit_x, right_fit_x, ploty


def sanity_checks(left_fit_x, right_fit_x, ploty):

    xm_per_pix = 3.7/700

    # Pick lowest points in the image for left and right sides and calculate distance
    left_min = min(left_fit_x) * xm_per_pix
    right_min = min(right_fit_x) * xm_per_pix

    left_max = max(left_fit_x) * xm_per_pix
    right_max = max(right_fit_x) * xm_per_pix

    # Pick highest points in the image for the left and right sides and calculate distance

    # If y coordinates are the same, the distance formula reduces to only x terms
    distance_low = np.sqrt((right_min - left_min)**2)
    distance_high = np.sqrt((right_max - left_max)**2)

    # lower threshold for lane width
    width_min_thr = 3.0
    # upper threshold for lane width
    width_max_thr = 3.8
    if ((distance_low > width_min_thr and distance_low < width_max_thr) or (distance_high > width_min_thr and distance_high < width_max_thr)):
        return True
    else:
        return False

    # print(left_min)
    # print(right_min)
    # print(distance_low)
    # print(left_max)
    # print(right_max)
    # print(distance_high)

    # Check if the lines are roughly parallel
    # Check similarity of curvature


def measure_curvature(ploty, left_x, right_x):

    # Conversion from pixel to real dimensions
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_fit = np.polyfit(ploty*ym_per_pix, left_x*xm_per_pix, 2)
    right_fit = np.polyfit(ploty*ym_per_pix, right_x*xm_per_pix, 2)
    print(left_fit)
    print(right_fit)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad, left_fit, right_fit


def draw_on_image(warped_img, left_fitx, right_fitx, ploty, Minv, undist, image, frame_bool):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    if frame_bool:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def process_image(image):

    # First the images are corrected for distortion
    undist = undistort_image(image, mtx, dist)

    # A sobel operator is applied as well as S channel is analyzed
    # in the HLS representation to identify lanes. The image is scaled
    # as binary
    combined_binary_img = create_threshold_binary_image(undist)

    # A trapezoid area is selected in front of the vehicle and warped
    # to bird's eye view
    transformed_img, Minv = perspective_transform(combined_binary_img)

    # if this is the first image, calculate polynomials from scratch i.e.
    # analyze the whole image, if polynomials are available, limit the search
    # area
    if left_fit_poly == [] and right_fit_poly == []:
        polyfit_img, left_fit_x, right_fit_x, ploty = fit_polynomial(transformed_img)

    else:
        left_fit_prev = left_fit_poly
        right_fit_prev = right_fit_poly
        polyfit_img, left_fit_x, right_fit_x, ploty = search_around_polynomial(transformed_img, left_fit_prev, right_fit_prev)

    # Perform sanity checks based on left and right lane points
    frame_bool = sanity_checks(left_fit_x, right_fit_x, ploty)

    if frame_bool:
        result = draw_on_image(transformed_img, left_fit_x, right_fit_x, ploty, Minv, undist, image, frame_bool)
        return result
    else:
        # For now, when the sanity check does not pass, an unprocessed undistored image is returned
        # THIS IS A TEMPORARY SOLUTION
        return undist

    # left_curve_radius, right_curve_radius, left_fit, right_fit = measure_curvature(ploty, left_fit_x, right_fit_x)
    # left_fit_array.append(left_fit)
    # right_fit_array.append(right_fit)

# ************** Start of Test Image Section ************** #

# Uncomment any one of the "test_img" line as well as the
# last three lines of this section to work with test images

# test_img = mpimg.imread('test_images\\straight_lines1.jpg')
# test_img = mpimg.imread('test_images\\straight_lines2.jpg')
test_img = mpimg.imread('test_images\\test1.jpg')
# test_img = mpimg.imread('test_images\\test2.jpg')
# test_img = mpimg.imread('test_images\\test3.jpg')
# test_img = mpimg.imread('test_images\\test4.jpg')
# test_img = mpimg.imread('test_images\\test5.jpg')
# test_img = mpimg.imread('test_images\\test6.jpg')

result = process_image(test_img)
plt.imshow(result)
plt.show()

# ************** End of Test Image Section **************** #

# Uncomment next 5 lines for the project video
# white_output = 'output_images\\project_video.mp4'
# clip1 = VideoFileClip('project_video.mp4')
# # new_clip = clip1.subclip(38, 45)
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)

# Uncomment next 4 lines for the challenge video
# Please note that the pipeline is not yet optimized for challenge videos
# white_output = 'output_images\\challenge_video.mp4'
# clip1 = VideoFileClip('challenge_video.mp4')
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)

# Uncomment next 4 lines for the harder challenge video
# Please note that the pipeline is not yet optimized for challenge videos
# white_output = 'output_images\\harder_challenge_video.mp4'
# clip1 = VideoFileClip('harder_challenge_video.mp4')
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)


