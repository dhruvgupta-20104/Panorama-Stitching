import glob
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        img_list = []

        for image in all_images:
            img = cv2.imread(image)
            img_list.append(img)

        homography_matrix_list = []

        base_img = img_list[-1]

        for img_index in range(len(img_list)-2, -1, -1):

            print("-------------------------------------------")

            right_img = base_img
            left_img = img_list[img_index]

            print("LOG: Stitching Image {} to the base image".format(img_index))
            
            kp_left, des_left = self.get_keypoints(left_img)
            kp_right, des_right = self.get_keypoints(right_img)

            print("LOG: Got Keypoints")

            matched_points = self.get_matched_points(kp_left, des_left, kp_right, des_right)

            print("LOG: Got Matched Points")

            homography_matrix = self.ransac(matched_points)
            homography_matrix_list.append(homography_matrix)

            print("LOG: Got Homography Matrix")

            right_image_shape = right_img.shape
            left_image_shape = left_img.shape

            left_image_corners = np.float32([[0, 0], [0, left_image_shape[0]], [left_image_shape[1], left_image_shape[0]], [left_image_shape[1], 0]]).reshape(-1, 1, 2)
            right_image_corners = np.float32([[0, 0], [0, right_image_shape[0]], [right_image_shape[1], right_image_shape[0]], [right_image_shape[1], 0]]).reshape(-1, 1, 2)

            left_image_corners = cv2.perspectiveTransform(left_image_corners, homography_matrix)

            list_of_points = np.concatenate((left_image_corners, right_image_corners), axis=0)

            [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

            translation_matrix = (np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])).dot(homography_matrix)

            output_img = cv2.warpPerspective(left_img, translation_matrix, (x_max-x_min, y_max-y_min))
            output_img[-y_min:right_image_shape[0]-y_min, -x_min:right_image_shape[1]-x_min] = right_img

            base_img = output_img

            print("LOG: Stitched Image {} to the base image".format(img_index))

            plt.imshow(base_img)
            plt.show()
        
        stitched_image = base_img
        print("-------------------------------------------")
        
        return stitched_image, homography_matrix_list 

    def get_keypoints(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        kp = sift.detect(img, None)
        kp, des = sift.compute(img, kp)
        return kp, des
    
    def get_matched_points(self, kp_left, des_left, kp_right, des_right):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        pre_matches = bf.knnMatch(des_left, des_right, k=2)

        matches = []
        for m, n in pre_matches:
            if m.distance < 0.75 * n.distance:
                matches.append([kp_left[m.queryIdx].pt, kp_right[m.trainIdx].pt])

        return matches

    
    def ransac(self, matches):
        most_inliers = 0
        best_homography_matrix = None
        threshold = 5
        num_trials = 5000
        sample_size = 4

        for i in range(num_trials):
            random_pts = random.choices(matches, k=sample_size)
            homography_matrix = self.calculate_homography_matrix(random_pts)
            num_inliers = 0
            for match in matches:
                left_image_point = np.array([match[0][0], match[0][1], 1])
                right_image_point = np.array([match[1][0], match[1][1], 1])
                predicted_right_image_point = np.dot(homography_matrix, left_image_point)
                predicted_right_image_point /= predicted_right_image_point[2]
                if np.linalg.norm(right_image_point - predicted_right_image_point) < threshold:
                    num_inliers += 1
            if num_inliers > most_inliers:
                most_inliers = num_inliers
                best_homography_matrix = homography_matrix
        return best_homography_matrix

    def calculate_homography_matrix(self, points):
        A = []
        for point in points:
            x, y = point[0]
            u, v = point[1]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H /= H[2, 2]

        return H