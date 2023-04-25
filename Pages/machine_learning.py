# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: machine_learning.py
@time: 2023/4/20 0:05
"""
# predict module
import cv2
import torch
import os
import os.path as osp
import joblib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from bisect import *

from deeplab import DeeplabV3
from retinaface import Retinaface
from model import BiSeNet
# st module
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO


# sklearn part
# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def vis_parsing_maps(im, parsing_anno, x, y, stride):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 255, 255], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [255, 255, 255], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    '''
    BGR模式
    part_colors = [[-],[面部（去除五官）],[右眉],[左眉],[-],[-],[-],[-],[-],[-],[鼻子],[-],[上嘴唇],[下嘴唇],[-],[-],[非人脸下部],[非人脸上部]]
    # 五官配置
    part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0], 
                [255, 0, 85], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0],
                [0, 0, 255], [0, 0, 0], [170, 0, 255],
                [0, 85, 255], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # 原始配置
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    '''

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    # print("vis_im: ", vis_im)
    vis_parsing_anno = parsing_anno.copy()
    vis_parsing_anno_color = np.zeros((im.shape[0], im.shape[1], 3)) + 0

    face_mask = np.zeros((im.shape[0], im.shape[1]))

    num_of_class = np.max(vis_parsing_anno)
    # print("num_of_class: ", num_of_class)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)  # 获得对应分类的的像素坐标

        idx_y = (index[0] + y).astype(np.int)
        idx_x = (index[1] + x).astype(np.int)

        # continue
        vis_parsing_anno_color[idx_y, idx_x, :] = part_colors[pi]  # 给对应的类别的掩码赋值
        # print(part_colors[pi])

        face_mask[idx_y, idx_x] = 1
        # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
        #     face_mask[idx_y,idx_x] = 0.35

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    face_mask = np.expand_dims(face_mask, 2)
    vis_im = vis_parsing_anno_color * face_mask + (1. - face_mask) * vis_im
    vis_im = vis_im.astype(np.uint8)

    return vis_im


def inference(img_size, image_path, model_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cpu()

    print('model : {}'.format(model_path))
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img_ = Image.open(image_path)
        img_ = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2BGR)
        img = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

        image = img.resize((img_size, img_size))
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cpu()
        out = net(img)[0]
        parsing_ = out.squeeze(0).cpu().numpy().argmax(0)

        # parsing_ = cv2.resize(parsing_, (img_.shape[1], img_.shape[0]), interpolation=cv2.INTER_NEAREST)

        parsing_ = parsing_.astype(np.uint8)
        vis_im = vis_parsing_maps(image, parsing_, 0, 0, stride=1)

    return vis_im


# 面积评级
def judgeLevel(score, breakpoints=[0.001, 0.1, 0.3, 0.5, 0.7, 0.9], levels='0123456'):
    i = bisect(breakpoints, score)
    return int(levels[i])


# hsv 直方图量化标准
hlist = [20, 40, 75, 155, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 316, 360]
svlist = [21, 98, 178, 255]


def quantilize(h, s, v):
    """hsv直方图量化"""
    # value : [21, 144, 23] h, s, v
    h = h * 2
    for i in range(len(hlist)):
        if h <= hlist[i]:
            h = i % 16
            # print("h: ", h)
            break
    for i in range(len(svlist)):
        if s <= svlist[i]:
            s = i
            # print("s: ", s)
            break
    for i in range(len(svlist)):
        if v <= svlist[i]:
            v = i
            # print("v: ", v)
            break

    return 9 * h + 3 * s + v


quantilize_ufunc = np.frompyfunc(quantilize, 3, 1)  # 自定义ufunc函数，即将quantilize函数转化为ufunc函数，其输入参数为３个，输出参数为１个。


def color_comment(img_hsv, maskROI, ptsROI):
    color_featrue = []
    # 一阶矩
    # 二阶矩
    mean, stddev = cv2.meanStdDev(img_hsv, mask=maskROI)
    h_mean = mean[0].tolist()[0]
    s_mean = mean[1].tolist()[0]
    v_mean = mean[2].tolist()[0]
    h_std = stddev[0].tolist()[0]
    s_std = stddev[1].tolist()[0]
    v_std = stddev[2].tolist()[0]
    # print(mean, stddev)
    # 三阶矩
    h = ptsROI[:, 0]
    s = ptsROI[:, 1]
    v = ptsROI[:, 2]
    h_offset = (np.mean(np.abs((h - h_mean) ** 3))) ** (1. / 3)
    s_offset = (np.mean(np.abs((s - s_mean) ** 3))) ** (1. / 3)
    v_offset = (np.mean(np.abs((v - v_mean) ** 3))) ** (1. / 3)
    # 获得颜色矩向量，将一阶矩归一化
    color_featrue.extend([h_mean, s_mean, v_mean, h_std, s_std, v_std, h_offset, s_offset, v_offset])
    # print("color_feature: ", color_featrue)
    return color_featrue


def colorsVector(image, maskROI):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pts = np.where(maskROI == 255)  # 获得ROI坐标
    ptsROI = hsv[pts[0], pts[1]]
    # print("len(ptsROI): ", len(ptsROI))
    nhsv = quantilize_ufunc(ptsROI[:, 0], ptsROI[:, 1], ptsROI[:, 2]).astype(np.uint8)  # 由于frompyfunc函数返回结果为对象，所以需要转换类型
    hist = cv2.calcHist([nhsv], [0], None, [148], [0, 148])  # 40x faster than np.histogram
    # print("hist.shape:", hist.shape)
    hist = hist.reshape(1, hist.shape[0]).astype(np.int32).tolist()[0]
    histVector = [item / 148 for item in hist]  # 归一化思考
    comment_vector = color_comment(hsv, maskROI, ptsROI)
    histVector.extend(comment_vector)
    # print("histVector", histVector)
    # print(len(histVector))
    # print(np.count_nonzero(histVector))

    return histVector


# app part
def app():
    deeplab = DeeplabV3()
    retinaface = Retinaface()
    name_classes = ["background", "red"]  # 背景，红斑
    st.markdown(
        '# <p style="background:#336699"  align=center><font face="微软雅黑" color="#FFFFFF" size="6"><b><br>毕业设计<br><br></b></font></p>',
        True)
    st.markdown("# <center> <u> PWS疗效高精度智能自动诊断系统 </u> </center> <br/> <br/>", True)

    st.markdown("## <center> 上传患者图片 </center>", True)
    upload = st.file_uploader("")

    cnt1, cnt2 = st.columns(2)  # 修改了
    if upload:
        with cnt1:
            st.markdown("### <center>上传图片</center>", True)
            st.image(upload)
        image = Image.open(upload)
        image = image.resize((512, 512), Image.ANTIALIAS)
        img_points = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img_orgin = img_points.copy()
        # 红斑分割
        r_image = deeplab.detect_image(image, count=False, name_classes=name_classes)  # 获得红斑区域（掩膜）
        img_size = 512  # 分辨率设置
        model_path = "D:/python_project/ST_app/weights/fp_512.pth"  # 模型路径
        normalSkin = inference(img_size=img_size, image_path=upload, model_path=model_path)  # 获得正常皮肤区域（掩膜）
        # ------------------------------------------------------------------------ #
        # 获得掩膜                                                                  #
        # ------------------------------------------------------------------------ #
        redROI_mask = r_image.convert("L")
        redROI_mask = cv2.cvtColor(np.asarray(redROI_mask), cv2.COLOR_RGB2BGR)  # PIL格式转opencv格式
        AllredGrayROI = cv2.cvtColor(redROI_mask, cv2.COLOR_RGB2GRAY)
        AllredcolorROI = cv2.bitwise_and(img_orgin, img_orgin, mask=AllredGrayROI)
        # print(redROI_mask.shape)
        ret, normalROI_mask = cv2.threshold(normalSkin, 254, 255, cv2.THRESH_BINARY)
        # ------------------------------------------------------------------------ #
        # 获取关键点                                                                #
        # ------------------------------------------------------------------------ #
        img_points = cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB)
        # ------------------------------------------------------------------------ #
        # 依据关键点绘图                                                             #
        # ------------------------------------------------------------------------ #
        point_list = retinaface.detect_image(img_points)

        r_image, roiList = retinaface.drawLine(img_points, point_list)
        # print(roiList)
        originROI1, originROI2, originROI3, originROI4 = retinaface.segmentROI(img_orgin, roiList)
        redROI1, redROI2, redROI3, redROI4 = retinaface.segmentROI(redROI_mask, roiList)  # 对红斑区域掩膜进行区域划分
        normalROI1, normalROI2, normalROI3, normalROI4 = retinaface.segmentROI(normalROI_mask,
                                                                               roiList)  # 对正常皮肤区域掩膜进行区域划分
        ROIs = [redROI1, redROI2, redROI3, redROI4, normalROI1, normalROI2, normalROI3, normalROI4]

        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        redROI1 = cv2.cvtColor(redROI1, cv2.COLOR_RGB2BGR)
        redROI2 = cv2.cvtColor(redROI2, cv2.COLOR_RGB2BGR)
        redROI3 = cv2.cvtColor(redROI3, cv2.COLOR_RGB2BGR)
        redROI4 = cv2.cvtColor(redROI4, cv2.COLOR_RGB2BGR)
        #
        normalROI1 = cv2.cvtColor(normalROI1, cv2.COLOR_RGB2BGR)
        normalROI2 = cv2.cvtColor(normalROI2, cv2.COLOR_RGB2BGR)
        normalROI3 = cv2.cvtColor(normalROI3, cv2.COLOR_RGB2BGR)
        normalROI4 = cv2.cvtColor(normalROI4, cv2.COLOR_RGB2BGR)

        redROIAreas = []
        normalROIAreas = []
        redcolorROIs = []
        # ------------------------------------------------------------------------ #
        # 面积评级                                                                  #
        # ------------------------------------------------------------------------ #
        for i in range(len(ROIs)):
            if i < 4:
                redGrayROI = cv2.cvtColor(ROIs[i], cv2.COLOR_RGB2GRAY)
                area = cv2.countNonZero(redGrayROI)
                redROIAreas.append(area)
                redcolorROI = cv2.bitwise_and(img_orgin, img_orgin, mask=redGrayROI)
                redcolorROIs.append(redcolorROI)

            else:
                normalGrayROI = cv2.cvtColor(ROIs[i], cv2.COLOR_RGB2GRAY)
                redGrayROI = cv2.cvtColor(ROIs[i - 4], cv2.COLOR_RGB2GRAY)
                normalGrayROI = cv2.add(normalGrayROI, redGrayROI)  # 红斑掩膜和正常皮肤掩膜做加和
                # print(normalGrayROI.shape)
                area = cv2.countNonZero(normalGrayROI)
                normalROIAreas.append(area)
        print("红斑各区域面积：", redROIAreas)
        print("正常皮肤各区域面积：", normalROIAreas)
        v = list(map(lambda x: x[0] / x[1], zip(redROIAreas, normalROIAreas)))
        # for i in v:
        #     print("area level is {:.2%} .".format(i))
        areaLevels = [judgeLevel(A) for A in v]
        print("面积评级：", areaLevels)
        # for i in areaLevels:
        #     print(i)
        # ------------------------------------------------------------------------ #
        # 颜色评级                                                                  #
        # ------------------------------------------------------------------------ #
        model_path = "D:\python_project\ST_app\weights\svm_class_colorVector.model"
        clf = joblib.load(model_path)  # 加载模型
        colorLevels = []
        i = 0
        for mask in ROIs[:4]:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img_input = cv2.bitwise_and(img_orgin, img_orgin, mask=mask)
            redROIVector = colorsVector(img_input, mask)
            nanVector = np.isnan(redROIVector)
            redROIVector = np.reshape(redROIVector, (1, -1))
            # print("redRIOVector: ", redROIVector)

            if True not in nanVector:
                preResult = clf.predict(redROIVector)
                colorLevels.append(int(preResult))
            else:
                preResult = 0
                colorLevels.append(preResult)
            i += 1

            # print("该区域颜色评级: ", preResult)

        print("颜色评级：", colorLevels)
        # ------------------------------------------------------------------------ #
        # FSASI评分                                                                 #
        # ------------------------------------------------------------------------ #
        Levels = [0.3 * areaLevels[i] * colorLevels[i] if i < 3 else 0.1 * areaLevels[i] * colorLevels[i] for i in
                  range(0, len(areaLevels))]
        FSASI = sum(Levels)
        print("FSASI评分为: ", FSASI)

        with cnt2:
            st.markdown("### <center>分割结果</center>", True)
            AllredcolorROI = Image.fromarray(cv2.cvtColor(AllredcolorROI, cv2.COLOR_BGR2RGB))
            st.image(AllredcolorROI)

        with st.container():
            # pws
            st.write('---')
            st.header('各区域鲜红斑痣图片')
            st.write('##')
            f_column, l_column, r_column, p_column = st.columns(4)
            with f_column:
                image_col, text_col = st.columns((1, 2))
                with image_col:
                    f_image = Image.fromarray(cv2.cvtColor(redcolorROIs[0], cv2.COLOR_BGR2RGB))
                    st.image(f_image)
                with text_col:
                    st.write(
                        """
                        前额部
                        """
                    )
                    st.write(
                        "面积评级：" + str(areaLevels[0])
                    )
                    st.write(
                        "颜色评级：" + str(colorLevels[0])
                    )
            with l_column:
                image_col, text_col = st.columns((1, 2))
                with image_col:
                    l_image = Image.fromarray(cv2.cvtColor(redcolorROIs[1], cv2.COLOR_BGR2RGB))
                    st.image(l_image)
                with text_col:
                    st.write(
                        """
                        左颧骨
                        """
                    )
                    st.write(
                        "面积评级：" + str(areaLevels[1])
                    )
                    st.write(
                        "颜色评级：" + str(colorLevels[1])
                    )
            with r_column:
                image_col, text_col = st.columns((1, 2))
                with image_col:
                    r_image = Image.fromarray(cv2.cvtColor(redcolorROIs[2], cv2.COLOR_BGR2RGB))
                    st.image(r_image)
                with text_col:
                    st.write(
                        """
                        右颧骨
                        """
                    )
                    st.write(
                        "面积评级：" + str(areaLevels[2])
                    )
                    st.write(
                        "颜色评级：" + str(colorLevels[2])
                    )
            with p_column:
                image_col, text_col = st.columns((1, 2))
                with image_col:
                    p_image = Image.fromarray(cv2.cvtColor(redcolorROIs[3], cv2.COLOR_BGR2RGB))
                    st.image(p_image)
                with text_col:
                    st.write(
                        """
                        口周
                        """
                    )
                    st.write(
                        "面积评级：" + str(areaLevels[3])
                    )
                    st.write(
                        "颜色评级：" + str(colorLevels[3])
                    )

        with st.container():
            st.write('---')
            st.header('PSI评分')
            st.write('##')
            st.write(str(FSASI))
            st.write('####')






