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


def convert_color_reverse(hsi_vector):
    convert_matrix = np.array([[1, -1 / 2, -1 / 2],
                               [0, math.sqrt(3) / 2, -math.sqrt(3) / 2],
                               [1 / 3, 1 / 3, 1 / 3]])
    reverse_matrix = np.linalg.inv(convert_matrix)
    hue, saturation, intensity = hsi_vector
    hue = math.radians(hue)
   #  print(hue, saturation, intensity)
    i_vector = np.array([0, 0, 0])
    i_vector[0] = math.sqrt((saturation ** 2) / (1 + math.tan(hue) ** 2))
    i_vector[1] = math.sqrt((saturation ** 2) * (math.tan(hue) ** 2) / (1 + math.tan(hue) ** 2))
    if 0 <= hue < math.pi/2:
        pass
    elif math.pi/2 <= hue < math.pi:
        i_vector[0] *= -1
    elif -math.pi/2 <= hue < 0:
        i_vector[1] *= -1
    elif -math.pi <= hue < -math.pi/2:
        i_vector[0] *= -1
        i_vector[1] *= -1

    i_vector[2] = intensity
    bgr_vector = np.dot(reverse_matrix, i_vector).astype(np.uint8)
    # print(bgr_vector)
    return bgr_vector


def extract_min_intensity_pixels(si):
    # si_sorted = sorted(si, key=lambda x:(x[0], x[1]))
    si_sorted = si[np.lexsort((si[:, 1], si[:, 0]))]
    '''col_num = 1
    si_sorted = si[np.argsort(si[:, col_num], kind='mergesort')]
    col_num = 0
    si_sorted = si_sorted[np.argsort(si_sorted[:, col_num], kind='mergesort')]'''

    # np.set_printoptions(threshold=np.inf)
    # print(si_sorted)
    s = 0
    extracted = np.empty((0, 2))
    for e in si_sorted:
        if e[0] > s:
            extracted = np.insert(extracted, 0, e, axis=0)
            s = e[0]
    # print(extracted)
    return extracted


def func1(param, x, y):
    residual = y - (param * x)
    return residual


def plot_specular(hsi, hue):
    hue_channel = hsi[:, :, 0]
    # ここで1次元配列になる
    same_hue_pixels = hsi[hue_channel == hue]
    if len(same_hue_pixels) > 0:
        min_intensity_extracted = extract_min_intensity_pixels(same_hue_pixels[:, 1:3])
        # saturation_array = same_hue_pixels[:, 1]
        # intensity_array = same_hue_pixels[:, 2]
        # print(same_hue_pixels)

        param0 = 0
        result = optimize.leastsq(func1, param0, args=(min_intensity_extracted[:, 0], min_intensity_extracted[:, 1]))
        # print(result)
        return result[0][0]
    return 0


def remove_specular_reflection(src):
    hsi_image = np.apply_along_axis(convert_color, 2, cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    hue_coefficient_list = [0 for _ in range(361)]
    # print(hsi_image[:, :, 0])
    for i in range(-180, 181):
        coefficient = plot_specular(hsi_image, i)
        hue_coefficient_list[i+180] = coefficient
    print(hue_coefficient_list)
    for y in range(hsi_image.shape[0]):
        for x in range(hsi_image.shape[1]):
           # print(hsi_image[y][x])
            if hue_coefficient_list[int(hsi_image[y][x][0])+180] > 0:
                hsi_image[y][x][2] = hue_coefficient_list[int(hsi_image[y][x][0])+180] * hsi_image[y][x][1]
    rgb_image = np.apply_along_axis(convert_color_reverse, 2, hsi_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


src = cv2.imread("orange.png")
dst = remove_specular_reflection(src)
cv2.imshow("", dst)

cv2.waitKey(0)