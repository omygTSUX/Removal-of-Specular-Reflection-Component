import cv2
import numpy as np
import math


def convert_color(bgr_vector):
    # print(bgr_vector)
    convert_matrix = np.array([[1, -1/2, -1/2],
                               [0, math.sqrt(3)/2, -math.sqrt(3)/2],
                               [1/3, 1/3, 1/3]])

    i_vector = np.dot(convert_matrix, bgr_vector)
    # print("i="+str(i_vector))
    hue = math.atan2(i_vector[1], i_vector[0])
    saturation = math.hypot(i_vector[0], i_vector[1])
    intensity = i_vector[2]

    return np.array([hue, saturation, intensity])


def remove_specular_reflection(src):
    hsi_image = np.apply_along_axis(convert_color, 2, src)
    print(hsi_image)
    return


g = np.zeros((1600, 1200, 3), dtype=np.uint8)
g[:, :, 2] = 255
remove_specular_reflection(g[:, :])
cv2.imshow("",g)

cv2.waitKey(0)