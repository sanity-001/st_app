import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils_retinaface import letterbox_image, preprocess_input
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


#------------------------------------#
#   请注意主干网络与预训练权重的对应
#   即注意修改model_path和backbone
#------------------------------------#
class Retinaface(object):
    _defaults = {
        #---------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择损失较低的即可。
        #---------------------------------------------------------------------#
        "model_path"        : 'D:\python_project\ST_app\logs\Retinaface_mobilenet0.25.pth',
        #---------------------------------------------------------------------#
        #   所使用的的主干网络：mobilenet、resnet50
        #---------------------------------------------------------------------#
        "backbone"          : 'mobilenet',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #---------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #---------------------------------------------------------------------#
        "input_shape"       : [1280, 1280, 3],
        #---------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   不同主干网络的config信息
        #---------------------------------------------------#
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        #---------------------------------------------------#
        #   先验框的生成
        #---------------------------------------------------#
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net    = RetinaFace(cfg=self.cfg, mode='eval').eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        '''
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        '''

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image = image.copy()
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            '''
            if self.cpu:
                self.anchors = self.anchors.cpu()
                image        = image.cpu()
            '''

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return old_image

            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            #---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            #---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # print(b[0], b[1], b[2], b[3], b[4])
            #---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            #---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            #---------------------------------------------------#
            #   将关键点进行连线
            #---------------------------------------------------#
            c_p = (int((b[5]+b[7])/2), int((b[6]+b[8])/2)) # 两眉中心点
            point_list = [(b[5], b[6]), (b[7], b[8]), c_p, (b[9], b[10]), (b[11], b[12]), (b[13], b[14])]
        return point_list
    
    def drawLine(self, old_image, pt_list):
        # print("point_list: ", pt_list)
        line_maps = {0:1, 2:3, 3:4, 5:3}
        finalLine = [] # 存放4组边界点，只使用第一组（两眉），第三组（鼻尖-左嘴角），第四组（鼻尖-右嘴角）
        for i in line_maps.keys():
            start_point = pt_list[i]
            end_point = pt_list[line_maps[i]]
            cv2.line(old_image, start_point, end_point, (0, 255, 0), 2)
            #---------------------------------------------------#
            #   获得边界关键点
            #---------------------------------------------------#
            w, h, _ = old_image.shape
            output = cv2.fitLine(points=np.array([start_point, end_point]), distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
            [vx, vy, x, y] = output
            # print(output)
            vx = float(vx)
            vy = float(vy)
            x = float(x)
            y = float(y)
            pts = []
            if vx == 0:
                X = lambda yy: x
                pts.append((int(X(0)), 0))
                pts.append((int(X(w)), w))
            elif vy == 0:
                Y = lambda xx: y
                pts.append((0, int(Y(0))))
                pts.append((h, int(Y(h))))
            else:
                # 从0遍历到height/width
                Y = lambda xx: vy / vx * (xx - x) + y  # 直线点斜式(k = vy / vx) -> y - y_0 = k(x - x_0) -> y = k(x - x_0) + y_0
                X = lambda yy: vx / vy * (yy - y) + x  # 直线点斜式(1 / k = vx / vy) -> (1 / k) * (y - y_0) = x - x_0 -> x =  (1 / k) * (y - y_0) + x_0
                # print(X(0)) X(0)代表直线y=0时对应的横坐标值，Y(0)代表直线当x=0时对应的纵坐标值
                # print(X(w))
                # 判断边界点
                if 0 <= int(X(0)) <= h:
                    pts.append((int(X(0)), 0))
                if 0 <= int(X(w)) <= h:
                    pts.append((int(X(w)), w))
                if 0 <= int(Y(0)) <= w:
                    pts.append((0, int(Y(0))))
                if 0 <= int(Y(h)) <= w:
                    pts.append((h, int(Y(h))))
            # print(pts)
            # print(len(pts))
            if len(pts) == 2:
                pt1 = pts[0]
                pt2 = pts[1]
                # cv2.line(old_image, pt1, pt2, (255, 255, 255), 3)
                finalLine.append(pt1)
                finalLine.append(pt2)
            else:
                pt1 = pts[0]
                pt2 = pts[-1]
                # cv2.line(old_image, pt1, pt2, (255, 255, 255), 3) 
                finalLine.append(pt1)
                finalLine.append(pt2) 
        
        extendPoints = [] # 存放左嘴角延长边界点和右嘴角延长线边界点
        for PointLine in finalLine[4:]:
            if PointLine[1] > pt_list[3][1]:
                extendPoints.append(PointLine)
        # print("extendPoints", extendPoints)

        cv2.line(old_image, pt_list[2], pt_list[3], (0, 255, 0), 2)  # 两眉中心点与鼻尖关键点连线
        cv2.line(old_image, finalLine[0], finalLine[1], (0, 255, 0), 2)  # 两眉关键点延长线
        cv2.line(old_image, extendPoints[0], pt_list[3], (0, 255, 0), 2)  # 鼻尖关键点与左嘴角关键点延长线
        cv2.line(old_image, pt_list[3], extendPoints[1], (0, 255, 0), 2)  # 鼻尖关键点与右嘴角关键点延长线

        segPtList = [finalLine[0], finalLine[1], pt_list[2], pt_list[3], extendPoints[0], extendPoints[1]]  # 两眉延长线左/右边界点，两眉中心点，鼻尖，左/右嘴角延长线边界点

        return old_image, segPtList
    
    def segmentROI(self, image, pts):
        #---------------------------------------------------#
        #   将图片进行区域划分(顺时针)
        #---------------------------------------------------#
        w, h = image.shape[:2]
        ROIpts1 = [(0, 0), (w, 0), pts[1], pts[0]]  # 额头区域
        ROIpts2 = [pts[0], pts[2], pts[3], pts[4]]  # 左脸区域
        ROIpts3 = [pts[5], pts[3], pts[2], pts[1]]  # 右脸区域
        ROIpts4 = [pts[4], pts[3], pts[5]]  # 下颚区域
        if pts[5][1] >= h:  # pts[5][1]先判断保证顺时针绘制ROI区域
            extraPt = [0, h]
            ROIpts2.append(extraPt)
            ROIpts4.append(extraPt) 
        
        if pts[4][1] >= h:
            extraPt = [w, h]
            ROIpts3.append(extraPt)
            ROIpts4.append(extraPt) 
        #---------------------------------------------------#
        #   绘制区域多边形
        #---------------------------------------------------#
        # 额头区域
        mask1 = np.zeros(image.shape[:2], np.uint8)
        ROIpts1 = np.array(ROIpts1)
        maskpts1 = ROIpts1.reshape((-1, 1, 2))
        mask1 = cv2.polylines(mask1, [maskpts1], True, 255)
        # 填充多边形
        resMask1 = cv2.fillPoly(mask1, [maskpts1], 255)
        ROI1 = cv2.bitwise_and(image, image, mask=resMask1)
        # 左脸区域
        mask2 = np.zeros(image.shape[:2], np.uint8)
        ROIpts2 = np.array(ROIpts2)
        maskpts2 = ROIpts2.reshape((-1, 1, 2))
        mask2 = cv2.polylines(mask2, [maskpts2], True, 255)
        resMask2 = cv2.fillPoly(mask2, [maskpts2], 255)
        ROI2 = cv2.bitwise_and(image, image, mask=resMask2)
        # 右脸区域
        mask3 = np.zeros(image.shape[:2], np.uint8)
        ROIpts3 = np.array(ROIpts3)
        maskpts3 = ROIpts3.reshape((-1, 1, 2))
        mask3 = cv2.polylines(mask3, [maskpts3], True, 255)
        resMask3 = cv2.fillPoly(mask3, [maskpts3], 255)
        ROI3 = cv2.bitwise_and(image, image, mask=resMask3)
        # 下颚区域
        mask4 = np.zeros(image.shape[:2], np.uint8)
        ROIpts4 = np.array(ROIpts4)
        maskpts4 = ROIpts4.reshape((-1, 1, 2))
        mask4 = cv2.polylines(mask4, [maskpts4], True, 255)
        resMask4 = cv2.fillPoly(mask4, [maskpts4], 255)
        ROI4 = cv2.bitwise_and(image, image, mask=resMask4)

        return ROI1, ROI2, ROI3, ROI4


    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   传入网络进行预测
                #---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                #-----------------------------------------------------------#
                #   对预测框进行解码
                #-----------------------------------------------------------#
                boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                #-----------------------------------------------------------#
                #   获得预测结果的置信度
                #-----------------------------------------------------------#
                conf    = conf.data.squeeze(0)[:, 1:2]
                #-----------------------------------------------------------#
                #   对人脸关键点进行解码
                #-----------------------------------------------------------#
                landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                #-----------------------------------------------------------#
                #   对人脸识别结果进行堆叠
                #-----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return np.array([])

            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        print("boxes_conf_landms: ", boxes_conf_landms)

        return boxes_conf_landms
