# -*- coding: utf-8 -*-
# @Time      : 2021/11/1 10:20 上午
# @Author    : Amos_Wang
# @File_Name : Digital images

# -- import Environment __
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# -- Function Definition --
# 定义卷积函数
def twodConv(image, box):
    """
    卷积函数
    :param image: 输入图像
    :param box: 输入卷积核
    :return: new_image
    """
    box = np.fliplr(np.flipud(box))
    box_row, box_col = box.shape
    image_row, image_col = image.shape
    new_row = image_row + box_row - 1
    new_col = image_col + box_col - 1
    add_row = int(box_row) // 2
    add_col = int(box_col) // 2
    # 零填充边界
    mid_image = np.zeros((new_row, new_col))
    new_image = np.zeros((image_row, image_col))
    # 复制原图
    mid_image[add_row:new_row - add_row, add_col:new_col - add_col] = image
    # 边界填充
    mid_image[0:add_row, add_col:new_col - add_col] = image[0, :]
    mid_image[new_row - add_row:, add_col:new_col - add_col] = image[-1, :]
    for i in range(add_col):
        mid_image[:, i] = mid_image[:, add_col]
        mid_image[:, new_col - 1 - i] = mid_image[:, new_col - 1 - add_col]
    # 卷积运算
    for i in range(image_row):
        for j in range(image_col):
            new_image[i, j] = np.sum(mid_image[i:i + box_row, j:j + box_col] * box)
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype(np.uint8)
    return new_image


# 引入随机噪声函数
def random_noise(image, noise_num):
    """
    添加随机噪点
    :param image: 需要加噪的图片
    :param noise_num: 添加的噪音点数目，一般是上千级别的
    :return: img_noise
    """

    img_noise = image
    rows, cols = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)  # 随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x, y] = 255
    return img_noise


# 引入椒盐噪声函数
def sp_noise(image, prob):
    """
    添加椒盐噪声
    :param image: 需要加噪的图片
    :param prob: 添加的噪声比例
    :return: img_noise
    """
    img_noise = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()  # 随机生成0-1之间的数字
            if rdn < prob:
                # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                img_noise[i][j] = 0
            elif rdn > thres:
                # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                img_noise[i][j] = 255
            else:
                # 其他情况像素点不变
                img_noise[i][j] = image[i][j]
    return img_noise


# 引入高斯噪声函数
def gauss_noise(image, mean=0, var=0.001):
    """
    添加高斯噪声
    :param image:原始图像
    :param mean : 均值
    :param var : 方差, 越大,噪声越大
    :return: img_noise
    """

    image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    img_noise = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if img_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    img_noise = np.clip(img_noise, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    img_noise = np.uint8(img_noise * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
    return img_noise


# 均值滤波函数
def my_blur(image, ksize):
    """
    均值滤波函数
    :param image: 原始图像
    :param ksize : 卷积核尺寸
    :return: img_blur
    """
    blur_box = 1 / 9 * np.ones(ksize)
    img_blur = twodConv(image, blur_box)
    return img_blur


# 中值滤波函数
def my_medianBlur(image, ksize):
    """
    中值滤波函数
    :param image: 原始图像
    :param ksize: 卷积核尺寸(一个数字)
    :return:
    """
    image_row, image_col = image.shape
    med_a = (ksize - 1) // 2
    new_image = image
    for i in range(med_a, image_row - med_a):
        for j in range(med_a, image_col - med_a):
            new_image[i, j] = np.median(image[i - med_a:i + med_a + 1, j - med_a:j + med_a + 1])
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype('uint8')
    return new_image


# 高斯滤波函数
def my_GaussianBlur(image, ksize, sigma=1.0):
    """
    高斯滤波函数
    :param image: 原始图像
    :param ksize: 卷积核尺寸
    :param sigma:
    :return: new_image
    """
    x, y = ksize
    add_x = int(x) // 2
    add_y = int(y) // 2
    gauss_box = np.zeros(ksize)
    for i in range(x):
        x_2 = (i - add_x) ** 2
        for j in range(y):
            y_2 = (j - add_y) ** 2
            H_ij = np.exp(-(x_2 + y_2) / (2 * (sigma ** 2)))
            gauss_box[i, j] = H_ij
    gauss_box = gauss_box / np.sum(gauss_box)
    new_image = twodConv(image, gauss_box)
    return new_image


# sobel 边缘检测
def my_sobel(image):
    """
    sobel 边缘检测
    :param image:
    :return: new_image
    """
    sobel_box = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
    new_image = twodConv(image, sobel_box)
    return new_image


# laplacian 锐化增强
def my_laplacian(image, model='four'):
    """
    laplacian 边缘增强
    :param image: 原始图像
    :param model: 检测方式："four" or "eight"
    :return: new_image, edge_image
    """
    if model == "four":
        lap_box = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])
    elif model == "eight":
        lap_box = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

    edge_image = twodConv(image, lap_box)
    new_image = edge_image + image
    new_image = new_image.clip(0, 255)
    return new_image, edge_image


# 实验内容1
def test_3_1and2():
    img_3_1 = cv.imread("3-1circuitBoard.tif")
    img_3_1 = cv.cvtColor(img_3_1, cv.COLOR_BGR2GRAY)

    img_3_1_noise_r = random_noise(img_3_1, 2000)  # 随机噪声实例化
    img_3_1_blur_cv = cv.blur(img_3_1_noise_r, (3, 3))  # cv均值滤波
    img_3_1_blur_my = my_blur(img_3_1_noise_r, (3, 3))  # 自构均值滤波

    img_3_1_noise_s = sp_noise(img_3_1, 0.1)  # 椒盐噪声实例化
    img_3_1_med_cv = cv.medianBlur(img_3_1_noise_s, 3)  # cv中值滤波
    img_3_1_med_my = my_medianBlur(img_3_1_noise_s, 3)  # 自构中值函数

    img_3_1_noise_g = gauss_noise(img_3_1, 0, 0.0001)  # 高斯噪声实例化
    img_3_1_gauss_cv = cv.GaussianBlur(img_3_1_noise_g, (3, 3), 1)  # cv高斯滤波
    img_3_1_gauss_my = my_GaussianBlur(img_3_1_noise_g, (3, 3), 1)  # 自构高斯滤波

    # cv.imshow("noise", img_3_1_noise)
    # cv.imshow("my_blur", img_3_1_XX_my)
    # cv.imshow("cv_blur", img_3_1_XX_cv)
    # cv.waitKey()
    # cv.destroyAllWindows()


# 实验内容3
def test_3_3():
    # sobel算子对图像边缘提取
    image_3_2 = cv.imread("3-2lena.jpg")
    image_3_2 = cv.cvtColor(image_3_2, cv.COLOR_BGR2GRAY)
    image_3_2_sobel_my = my_sobel(image_3_2)
    image_3_2_sobel_cv = cv.Sobel(image_3_2, cv.CV_16S, 1, 1)
    image_3_2_sobel_cv = cv.convertScaleAbs(image_3_2_sobel_cv)

    image_3_2_my_compare = np.hstack([image_3_2, image_3_2_sobel_my])
    image_3_2_cv_compare = np.hstack([image_3_2, image_3_2_sobel_cv])
    cv.imwrite("3-2自构sobel函数.png", image_3_2_my_compare)
    cv.imwrite("3-2cv_sobel函数.png", image_3_2_cv_compare)


# 实验内容4
def test_3_4():
    image_3_3 = cv.imread("3-3moon.tif")
    image_3_3 = cv.cvtColor(image_3_3, cv.COLOR_BGR2GRAY)

    # laplacian 锐化增强
    image_3_3_lap_my, image_3_3_edge_my = my_laplacian(image_3_3)
    image_3_3_edge_cv = cv.Laplacian(image_3_3, cv.CV_64F)
    image_3_3_edge_cv = cv.convertScaleAbs(image_3_3_edge_cv)

    # 高提升滤波
    image_3_3_gauss = my_GaussianBlur(image_3_3, (5, 5), 0.8)
    image_3_3_mask = image_3_3 - image_3_3_gauss
    image_3_3_higher = image_3_3 + 2 * image_3_3_mask
    image_3_3_higher = image_3_3_higher.clip(0, 255)

    cv.imwrite("higher.tif", image_3_3_higher)
    cv.imwrite("laplacian.tif", image_3_3_lap_my)
    cv.imwrite("laplacian_edge_my.tif", image_3_3_edge_my)
    cv.imwrite("laplacian_edge_cv.tif", image_3_3_edge_cv)


# -- Main Function --
if __name__ == "__main__":
    flag = input("输入实验序号1-4：")
    if flag == "1" or "2":
        test_3_1and2()  # 实验内容1&2
    elif flag == "3":
        test_3_3()  # 实验内容3
    elif flag == "4":
        test_3_4()  # 实验内容4
    print("Finish my homework, by Zhehan Wang")

# 废案
'''
def conv(image, box):
    image_row, image_col = image.shape
    box = np.fliplr(np.flipud(box))
    box_row, box_col = box.shape
    row_a = (box_row - 1) // 2
    col_a = (box_col - 1) // 2
    new_image = image
    for i in range(row_a, image_row - row_a):
        for j in range(col_a, image_col - col_a):
            new_image[i, j] = np.sum(image[i - row_a:i + row_a + 1, j - col_a:j + col_a + 1] * box)
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype('uint8')
    return new_image
'''
