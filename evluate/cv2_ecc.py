#!/usr/bin/python
"""
    OpenCV Image Alignment Example
    Copyright 2015 by Satya Mallick <spmallick@learnopencv.com>
"""

import cv2
import numpy as np
import time
import skimage.io as io

def estimate_affine_ecc(im1,im2):
    # Read the images to be aligned
    # im1 = cv2.imread("image1.jpg")
    # im2 = cv2.imread("image2.jpg")

    # im2 = cv2.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_31.png')
    # im1 = cv2.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_17.png')

    im1_gray = cv2.cvtColor(np.asarray(im1), cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(np.asarray(im2), cv2.COLOR_RGB2GRAY)

    # Convert images to grayscale
    # im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    im1_size = im1.shape

    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_AFFINE
    # warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Enhanced Correlation Coefficient (ECC)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    start = time.time()
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (im1_size[1], im1_size[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (im1_size[1], im1_size[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    end = time.time()
    print('Alignment time (s): ', end - start)
    # print(warp_matrix)
    # # Show final results
    # cv2.imshow("Image 1", im1)
    # cv2.imshow("Image 2", im2)
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)


if __name__ == '__main__':
    im1 = io.imread('../datasets/row_data/multispectral/mul_1t_s.png')
    im2 = io.imread('../datasets/row_data/multispectral/mul_1s_s.png')

    estimate_affine_ecc(im1,im2)
