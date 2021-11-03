# -*- coding: utf-8 -*-
# @Time      : 2021/10/18 10:12 上午
# @Author    : Amos_Wang
# @File_Name : Digital images

# -- import Environment __
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# -- Function Definition --
# 直方图均衡化函数 实验内容1所需函数
def hist_equal_my(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])  # 获取直方图信息
    img_dst = img.copy()  # 复制原图获得目标最终图像
    pixels_sum = img.size  # 得到总像素点数
    # 构建中间变量
    img_mid = np.zeros(256)
    img_mid_S_k = np.zeros(256)

    j = 0
    for i in hist:
        # 计算s_k
        if j > 0:
            img_mid[j] = i / pixels_sum + img_mid[j - 1]
        else:
            img_mid[j] = i / pixels_sum
        # 非归一化S(k)
        img_mid_S_k[j] = round(img_mid[j] * 255)
        j = j + 1
    # 图像重映射
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_dst[i][j] = img_mid_S_k[img[i][j]]
    return img_dst


# 实验内容1
def test_2_1():
    # 获取图片
    img2_1 = cv.imread("2.1.tif")
    img2_1 = cv.cvtColor(img2_1, cv.COLOR_BGR2GRAY)
    # opencv直方图均衡
    img2_1_dst_cv = cv.equalizeHist(img2_1)
    # 自构函数直方图均衡
    img2_1_dst_my = hist_equal_my(img2_1)
    # 绘制直方图
    hist2_1 = cv.calcHist([img2_1], [0], None, [256], [0, 256])
    hist2_1_dst_cv = cv.calcHist([img2_1_dst_cv], [0], None, [256], [0, 256])
    hist2_1_dst_my = cv.calcHist([img2_1_dst_my], [0], None, [256], [0, 256])

    # 图片展示区域
    plt.subplot(3, 2, 1)
    plt.imshow(img2_1, cmap='gray')
    plt.title('src')
    plt.subplot(3, 2, 2)
    plt.plot(hist2_1, label="imgSrc", color="b")
    plt.subplot(3, 2, 3)
    plt.imshow(img2_1_dst_cv, cmap='gray')
    plt.title('hist_cv')
    plt.subplot(3, 2, 4)
    plt.plot(hist2_1_dst_cv, label="imgHist_cv", color="r")
    plt.subplot(3, 2, 5)
    plt.imshow(img2_1_dst_my, cmap='gray')
    plt.title('hist_my')
    plt.subplot(3, 2, 6)
    plt.plot(hist2_1_dst_my, label="imgHist_my", color="g")
    plt.savefig("result.PNG")
    plt.show()

# # 实验内容2
# def test_2_2():


# -- Main Function --
if __name__ == "__main__":
    # flag = input("输入实验序号1-4：")
    # if flag == "1":
    #     test_2_1()  # 实验内容1
    # elif flag == "2":
    #     test_2_2()  # 实验内容2
    test_2_1()  # 实验内容1
    print("Finish my homework, by Zhehan Wang")
