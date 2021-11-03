# -*- coding: utf-8 -*-
# @Time      : 2021/11/1 17:50
# @Author    : Amos_Wang
# @File_Name : Digital images

# -- import Environment __
import cv2 as cv
import numpy as np

# -- Function Definition --


# -- Main Function --
if __name__ == "__main__":
    img3_4 = cv.imread("3-4bonescan.tif")
    img3_4 = cv.cvtColor(img3_4, cv.COLOR_BGR2GRAY)
    img3_4_mask_x = cv.Sobel(img3_4, -1, 1, 0)
    img3_4_mask_x = cv.convertScaleAbs(img3_4_mask_x)
    img3_4_mask_y = cv.Sobel(img3_4, -1, 0, 1)
    img3_4_mask_y = cv.convertScaleAbs(img3_4_mask_y)
    img3_4_mask = cv.addWeighted(img3_4_mask_x, 0.5, img3_4_mask_y, 0.5, 0)
    img3_4_add = img3_4 + img3_4_mask
    img3_4_add = img3_4_add.clip(0, 255)

    # cv.imshow("src", img3_4)
    cv.imshow("mask", img3_4_mask)
    # cv.imshow("add", img3_4_add)
    cv.waitKey()
    cv.destroyAllWindows()
    print("Hello World")
