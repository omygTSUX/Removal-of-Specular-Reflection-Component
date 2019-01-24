import cv2
import numpy as np
import math
from scipy import optimize


def convert_color(bgr_vector):
    # print(bgr_vector)
    convert_matrix = np.array([[1, -1/2, -1/2],
                               [0, math.sqrt(3)/2, -math.sqrt(3)/2],
                               [1/3, 1/3, 1/3]])

    i_vector = np.dot(convert_matrix, bgr_vector)
    # print("i="+str(i_vector))
    hue = int(math.degrees(math.atan2(i_vector[1], i_vector[0])))
    saturation = math.hypot(i_vector[0], i_vector[1])
    intensity = i_vector[2]

    return np.array([hue, saturation, intensity])


def extract_min_intensity_pixels():
    return



def plot_specular(hsi, hue):
    hue_channel = hsi[:, :, 0]
    # ここで1次元配列になる
    same_hue_pixels = hsi[hue_channel == hue]
    saturation_array = same_hue_pixels[:, 1]
    intensity_array = same_hue_pixels[:, 2]
    print(same_hue_pixels)

    def func1(param, x, y):
        residual = y - (param[0] * x + param[1])
        return residual

    param = [0, 0]
    print(optimize.leastsq(func1, param, args=(saturation_array, intensity_array)))


def remove_specular_reflection(src):
    hsi_image = np.apply_along_axis(convert_color, 2, cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    # print(hsi_image[:, :, 0])
    plot_specular(hsi_image, hsi_image[250][250][0])
    return


g = np.zeros((16, 12, 3), dtype=np.uint8)
g[:, :, 1] = 255
src = cv2.imread("pokeball.jpg")
remove_specular_reflection(src)
cv2.imshow("", g)

cv2.waitKey(0)