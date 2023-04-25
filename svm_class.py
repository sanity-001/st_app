# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: svm_class.py
@time: 2023/3/19 22:48
"""

import numpy as np
import os
import cv2
import time
from sklearn import svm
import joblib


def read_all_data(file_path):
    cName = ['1', '2', '3', '4']
    # 得到一个图像文件名列表flist
    i = 0
    for c in cName:
        train_data_path = os.path.join(file_path, c)
        # 获取文件夹下的所有图片路径列表
        flist_ = get_file_list(train_data_path)

        if i == 0:
            dataMat, dataLabel = read_and_convert(flist_)
        else:
            dataMat_, dataLabel_ = read_and_convert(flist_)
            # 按轴axis0连接array组成一个新的array
            dataMat = np.concatenate((dataMat, dataMat_), axis=0)
            dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
            # print(dataMat.shape)
        # print(len(dataLabel))
        i += 1

    return dataMat, dataLabel


def read_and_convert(imgFileList):
    dataLabel = []  # 存放类标签
    # 计算图像个数
    dataNum = len(imgFileList)
    # dataNum * 512 * 512 * 3 的矩阵
    # 颜色矩9个元素
    dataMat = np.zeros((dataNum, 9))
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        # 得到 类标签 如  B_5.jpg
        imgClass = os.path.split(imgFileList[i])
        classTag = os.path.basename(imgClass[0])  # 获取路径最后一个文件名
        dataLabel.append(classTag)
        dataMat[i, :] = color_comment(imgNameStr)

    return dataMat, dataLabel


def get_file_list(path):
    file_list = []
    # 获取path路径下的所有文件名
    for file_name in os.listdir(path):
        fin_path = os.path.join(path, file_name)
        if fin_path.endswith('.jpg'):
            file_list.append(fin_path)

    return file_list


# 将图像转换为向量
def img2vector(imgFile):
    img = cv2.imread(imgFile, 1)
    img_arr = np.array(img)
    # 对图像进行归一化
    img_normlization = img_arr / 255
    # 1 * 1875 向量
    img_arr2 = np.reshape(img_normlization, (1, -1))
    return img_arr2


# 计算图像颜色矩
def color_comment(img):
    img = cv2.imread(img)
    b, g, r = cv2.split(img)
    b = b[np.where(b >= 5)]
    g = g[np.where(g >= 5)]
    r = r[np.where(r >= 5)]
    color_featrue = []
    # 一阶矩
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    # 二阶矩
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    # 三阶矩
    r_offset = (np.mean(np.abs((r - r_mean) ** 3))) ** (1. / 3)
    g_offset = (np.mean(np.abs((g - g_mean) ** 3))) ** (1. / 3)
    b_offset = (np.mean(np.abs((b - b_mean) ** 3))) ** (1. / 3)
    color_featrue.extend([r_mean, g_mean, b_mean, r_std, g_std, b_std, r_offset, g_offset, b_offset])
    return color_featrue


def create_svm(dataMat, dataLabel, path):
    clf = svm.SVC(C=1.0, kernel='rbf')
    # 开始训练模型
    rf = clf.fit(dataMat, dataLabel)
    # 存储训练好的模型
    joblib.dump(rf, path)
    return clf


if __name__ == "__main__":

    st = time.clock()
    path = "C:/Users/Administrator/Desktop/roi_class/origin"
    dataMat, dataLabel = read_all_data(path)
    model_path = os.path.join(path, 'svm_class_colorComment.model')
    create_svm(dataMat, dataLabel, model_path)
    et = time.clock()
    print("Training spent {:.4f}s.".format((et - st)))