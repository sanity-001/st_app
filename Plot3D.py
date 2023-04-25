# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: Plot3D.py
@time: 2023/4/25 17:39
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('C:/Users/Administrator/Desktop/2023-04-25 174108.png')
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
b, g, r = cv.split(img)
img = cv.merge([r, g, b]) #为了适应fc中元组顺序为RGB，故调整
draw = np.reshape(img, (-1, 3))
lists = list(set([tuple(t) for t in draw]))  #去除列表中重复行
print(np.array(lists))
draw = tuple(draw/255)
ax = plt.subplot(1, 1, 1, projection='3d')  # 创建一个三维绘图，并设置颜色

ax.scatter(r, g, b, s=1, color=draw)  # 以RGB值为坐标绘制数据点，并以RGB值为点颜色

ax.set_zlim(0, 255)
ax.set_ylim(0, 255)
ax.set_xlim(0, 255)
ax.set_zlabel('H', c='r')  # 坐标轴
ax.set_ylabel('S', c='g')
ax.set_xlabel('V', c='b')
ax.set_title('clolored image distribution', color='w')
ax.tick_params(labelcolor='w')
plt.show()
