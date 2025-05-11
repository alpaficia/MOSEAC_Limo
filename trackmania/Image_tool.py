# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:15:18 2023

@author: Dong
"""

import cv2
import skimage
import numpy as np
from sklearn import linear_model


class ImageTool:
    def __init__(self):
        self.edge_low_param = 100
        self.edge_up_param = 200
        self.size = (4, 4)
        self.len_of_image_history = 4
        self.img_height = 64
        self.polygons_line = np.array([[(0, self.img_height), (0, int(0.6 * self.img_height)), (25, 20), (35, 20), (64, int(0.6 * self.img_height)), (64, self.img_height)]])
        self.polygons_car = np.array([[(10, int(0.9 * self.img_height)), (15, int(0.6 * self.img_height)), (39, int(0.6 * self.img_height)), (54, int(0.9 * self.img_height))]])
        self.ransac = linear_model.RANSACRegressor()

    def edge_detection(self, images):
        img_edge = np.zeros([self.len_of_image_history, 64, 64])
        
        for i in range(self.len_of_image_history):
            edge = cv2.Canny(images[i], self.edge_low_param, self.edge_up_param)
            edge_segment_line = self.do_segment(self.polygons_line, edge)
            edge_segment_car = self.do_segment(self.polygons_car, edge)
            left_line, right_line = self.fit_lines(edge_segment_line)
            line_gray = self.visualize_lines(images[i], left_line, right_line)
            
            if line_gray is not None:
                img_edge[i] = cv2.addWeighted(edge_segment_car, 1, line_gray, 1, 0)
            else:
                img_edge[i] = edge_segment_car
        
        return img_edge

    def do_segment(self, polygons, frame):
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, polygons, 255)
        segment = cv2.bitwise_and(frame, mask)
        return segment

    def fit_lines(self, edge_segment_line):
        hough = cv2.HoughLinesP(edge_segment_line, 0.8, np.pi / 180, 15)
        left_x, left_y, right_x, right_y = self.calculate_line_groups(hough)
        left_line = self.fit_and_predict(left_x, left_y)
        right_line = self.fit_and_predict(right_x, right_y)
        return left_line, right_line

    def fit_and_predict(self, x, y):
        self.ransac.fit(x, y)
        X = np.arange(x.min(), x.max())[:, np.newaxis]
        y_predict = self.ransac.predict(X)
        y_predict = y_predict.astype(int)
        return np.hstack((X, y_predict))

    def calculate_line_groups(self, lines):
        if lines is None:
            return None

        lines = np.array(lines).reshape(-1, 4)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # Calculate slopes
        slopes = (y1 - y2) / (x1 - x2 + 1e-32)

        left_mask = slopes < 0
        right_mask = slopes >= 0

        left_x = np.concatenate((x1[left_mask], x2[left_mask]))
        left_y = np.concatenate((y1[left_mask], y2[left_mask]))
        right_x = np.concatenate((x1[right_mask], x2[right_mask]))
        right_y = np.concatenate((y1[right_mask], y2[right_mask]))

        return left_x.reshape(-1, 1), left_y.reshape(-1, 1), right_x.reshape(-1, 1), right_y.reshape(-1, 1)

    def visualize_lines(self, frame, line_left, line_right):
        if line_left is None and line_right is None:
            return None

        lines_visualize = np.zeros_like(frame)

        if line_left is not None:
            for i in range(len(line_left) - 1):
                cv2.line(lines_visualize, (line_left[i][0], line_left[i][1]), (line_left[i + 1][0], line_left[i + 1][1]), (255, 255, 255), 2)

        if line_right is not None:
            for i in range(len(line_right) - 1):
                cv2.line(lines_visualize, (line_right[i][0], line_right[i][1]), (line_right[i + 1][0], line_right[i + 1][1]), (255, 255, 255), 2)

        line_gray = cv2.cvtColor(lines_visualize, cv2.COLOR_BGR2GRAY)
        return line_gray
    
    def image_unified(self, image):
        return image / 255.0  # converage from (0, 255) to (0, 1)
    
    def image_average_pooling(self, image):
        img_pooling = np.zeros([self.len_of_image_history, 16, 16])
        for i in range(self.len_of_image_history):
            img_pooling[i] = skimage.measure.block_reduce(image[i], self.size, np.mean)
        return img_pooling
    
    def image_reshape_unify(self, image):
        return image.reshape(1024,)
