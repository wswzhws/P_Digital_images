# -*- coding: utf-8 -*-
# @Time      : 2021/10/6 1:44 下午
# @Author    : Amos_Wang
# @File_Name : Digital images

# -- import Environment __
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


# -- Function Definition --

# test_1_2所需函数(改变空间、灰度分辨率)
def change_pixel(img):
    x, y = img.shape  # 读出lena的像素大小
    img_25 = img[1:x:5, 1:y:5]  # 改变空间分辨率25倍
    img_100 = img[1:x:10, 1:y:10]  # 改变空间分辨率100倍
    img_2_7 = img // 2 * 2  # 改变256Bit -> 128Bit
    img_2_4 = img // 16 * 16  # 改变256Bit -> 16Bit
    img_Couple = [img, img_25, img_100, img, img_2_7, img_2_4]
    return img_Couple


# test_1_2所需函数(展示图象)
def show_couple(img_couple):
    titles = ['Original Image', 'Space_25', 'Space_100', 'Original Imag', 'Gray_128', 'Gray_16']
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_couple[i], cmap='gray')  # 没有cmap=“gray”会出现色差
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


# test_1_3 所需函数(最近邻插值)
def my_resize_near(imgsrc, fx, fy):
    rowSrc, colSrc = imgsrc.shape

    rowDst = int(rowSrc * fy)  # 目标y值
    colDst = int(colSrc * fx)  # 目标x值
    imgDst = np.zeros((rowDst, colDst), dtype='uint8')

    fy = rowSrc / rowDst  # 实际缩放系数
    fx = colSrc / colDst  # 实际缩放系数

    for i in range(rowDst):
        for j in range(colDst):
            srcX = int(j * fx)
            srcY = int(i * fy)
            imgDst[i, j] = imgsrc[srcY, srcX]

    return imgDst


# test_1_3 所需函数(双线性插值)
def my_resize_liner(imgsrc, fx, fy):
    rowSrc, colSrc = imgsrc.shape

    rowDst = int(rowSrc * fy)  # 目标y值
    colDst = int(colSrc * fx)  # 目标x值
    imgDst = np.zeros((rowDst, colDst), dtype='uint8')

    fy = rowSrc / rowDst  # 实际缩放系数
    fx = colSrc / colDst  # 实际缩放系数

    for i in range(rowDst):
        for j in range(colDst):
            # 将防缩前后两个图象的几何中心重合，确保边缘像素参与计算
            srcX = float((j + 0.5) * fx - 0.5)
            srcY = float((i + 0.5) * fy - 0.5)

            # 向下取整，代表靠近源点的左上角的那一点的行列号
            srcXint = math.floor(srcX)
            srcYint = math.floor(srcY)

            # 取出小数部分，用于构造权值
            srcXfloat = srcX - srcXint
            srcYfloat = srcY - srcYint

            if srcXint + 1 == colSrc or srcYint + 1 == rowSrc:
                imgDst[i, j] = imgsrc[srcYint, srcXint]
                continue
            imgDst[i, j] = (1. - srcYfloat) * (1. - srcXfloat) * imgsrc[srcYint, srcXint] + \
                           (1. - srcYfloat) * srcXfloat * imgsrc[srcYint, srcXint + 1] + \
                           srcYfloat * (1. - srcXfloat) * imgsrc[srcYint + 1, srcXint] + \
                           srcYfloat * srcXfloat * imgsrc[srcYint + 1, srcXint + 1]

    return imgDst


# test_1_3 所需函数(旋转+最近邻)
def my_resize_near_revolve(imgsrc, angle_rad):
    rowSrc, colSrc = imgsrc.shape

    # 自己构建旋转矩阵
    # A = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
    #               [math.sin(angle_rad),  math.cos(angle_rad)]])

    # 获得生成图片尺寸大小
    rowDst = math.ceil(colSrc * abs(math.sin(angle_rad)) + abs(rowSrc * math.cos(angle_rad)))
    colDst = math.ceil(colSrc * abs(math.cos(angle_rad)) + abs(rowSrc * math.sin(angle_rad)))
    imgDst = np.zeros((rowDst, colDst), dtype='uint8')

    # 获得旋转偏移量
    rowDrift = -colSrc * math.sin(angle_rad) * math.cos(angle_rad)
    colDrift =  colSrc * math.sin(angle_rad) * math.sin(angle_rad)

    for i in range(rowDst):
        for j in range(colDst):
            srcY = math.floor(j * math.sin(angle_rad) + i * math.cos(angle_rad) + rowDrift)
            srcX = math.floor(j * math.cos(angle_rad) - i * math.sin(angle_rad) + colDrift)
            if srcX <= 0 or srcX >= colSrc or srcY <= 0 or srcY >= rowSrc:
                imgDst[i, j] = 0
            else:
                imgDst[i, j] = imgsrc[srcY, srcX]

    return imgDst


# test_1_3 所需函数(旋转+双线性)
def my_resize_liner_revolve(imgsrc, angle_rad):
    rowSrc, colSrc = imgsrc.shape

    # 获得生成图片尺寸大小
    rowDst = math.ceil(colSrc * abs(math.sin(angle_rad)) + abs(rowSrc * math.cos(angle_rad)))
    colDst = math.ceil(colSrc * abs(math.cos(angle_rad)) + abs(rowSrc * math.sin(angle_rad)))
    imgDst = np.zeros((rowDst, colDst), dtype='uint8')

    # 获得旋转偏移量
    rowDrift = -colSrc * math.sin(angle_rad) * math.cos(angle_rad)
    colDrift = colSrc * math.sin(angle_rad) * math.sin(angle_rad)

    for i in range(rowDst):
        for j in range(colDst):
            srcY = float(j * math.sin(angle_rad) + i * math.cos(angle_rad) + rowDrift)
            srcX = float(j * math.cos(angle_rad) - i * math.sin(angle_rad) + colDrift)

            srcXint = math.floor(srcX)
            srcYint = math.floor(srcY)

            # 取出小数部分，用于构造权值
            srcXfloat = srcX - srcXint
            srcYfloat = srcY - srcYint

            if srcXint <= 0 or srcXint >= colSrc or srcYint <= 0 or srcYint >= rowSrc \
            or srcXint+1 <= 0 or srcXint+1 >= colSrc or srcYint+1 <= 0 or srcYint+1 >= rowSrc:
                imgDst[i, j] = 0
                continue
            imgDst[i, j] = (1. - srcYfloat) * (1. - srcXfloat) * imgsrc[srcYint, srcXint] + \
                           (1. - srcYfloat) * srcXfloat * imgsrc[srcYint, srcXint + 1] + \
                           srcYfloat * (1. - srcXfloat) * imgsrc[srcYint + 1, srcXint] + \
                           srcYfloat * srcXfloat * imgsrc[srcYint + 1, srcXint + 1]

    return imgDst


# 实验内容1
def test_1_1():
    img_Tina_RGB = cv.imread("IMG_1113.JPG")  # 读取彩色图像,需注意，这里严格意义是BGR 而不是 RGB
    img_Tina_Black = cv.cvtColor(img_Tina_RGB, cv.COLOR_BGR2GRAY)  # 读取灰度图像
    # img_Tina_ind  读取索引图像(opencv中未找到索引图像概念)

    cv.imshow("Black", img_Tina_Black)  # 显示灰度图像
    cv.imshow("RGB", img_Tina_RGB)  # 显示RGB图像
    cv.waitKey(0)  # 按下任意键退出
    cv.destroyAllWindows()  # 关闭所有窗口

    cv.imwrite("Tina_Black.JPG", img_Tina_Black)  # 保存灰度图像


# 实验内容2
def test_1_2():
    img_Lena = cv.imread("lena2.BMP", 0)  # 读取lena的图像
    img_Photograph = cv.imread("cman.BMP", 0)  # 读取摄影师图像
    img_Crowd = cv.imread("crowd.BMP", 0)  # 读取人群图像

    # Lena、Photograph、Crowd 相关图片
    img_Lena_Couple = change_pixel(img_Lena)
    img_Photograph_Couple = change_pixel(img_Photograph)
    img_Crowd_Couple = change_pixel(img_Crowd)

    show_couple(img_Lena_Couple)
    show_couple(img_Photograph_Couple)
    show_couple(img_Crowd_Couple)


# 实验内容3
def test_1_3():
    img_Crowd = cv.imread("crowd.BMP", 0)  # 读取人群图像

    # opencv自带插值函数
    # img_Crowd_Near_CV = cv.resize(img_Crowd, None, fx=1.5, fy=1.5, interpolation=cv.INTER_NEAREST)  # 最近邻插值
    # img_Crowd_Liner_CV = cv.resize(img_Crowd, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)  # 双线性插值
    # img_Crowd_Cubic_CV = cv.resize(img_Crowd, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)  # 双三次内插

    # 自己构造内插函数
    # img_Crowd_Near_my = my_resize_near(img_Crowd, fx=1.5, fy=1.5)
    # img_Crowd_Liner_my = my_resize_liner(img_Crowd, fx=1.5, fy=1.5)

    # 自己构建旋转函数
    img_Crowd_near_revolve = my_resize_near_revolve(img_Crowd, math.pi / 6)
    img_Crowd_liner_revolve = my_resize_liner_revolve(img_Crowd, math.pi / 6)

    # 展示图片
    # cv.imshow("img", img_Crowd)
    # cv.imwrite("near.JPG", img_Crowd_Near_my)
    # cv.imwrite('liner.JPG', img_Crowd_Liner_my)
    cv.imwrite("near_revolve.JPG", img_Crowd_near_revolve)
    cv.imwrite("liner_revolve.JPG", img_Crowd_liner_revolve)
    # cv.imwrite("near_cv.JPG", img_Crowd_Near_CV)
    # cv.imwrite("liner_cv.JPG", img_Crowd_Liner_CV)
    # cv.imwrite("cubic_cv.JPG", img_Crowd_Cubic_CV)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


# -- Main Function --
if __name__ == "__main__":
    flag = input("输入实验序号1-3：")
    if flag == "1":
        test_1_1()  # 实验内容1
    elif flag == "2":
        test_1_2()  # 实验内容2
    elif flag == "3":
        test_1_3()  # 实验内容3
    print("Finish my homework, by Zhehan Wang")
