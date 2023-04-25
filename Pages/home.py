# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: home.py
@time: 2023/4/20 0:05
"""

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app():
    load_css('style/style.css')
    lottie_coding = load_lottie("https://assets1.lottiefiles.com/packages/lf20_F9aiyX22UC.json")
    img_sphere = Image.open('images/PWS.jpg')
    img_phase_separation = Image.open('images/FSASI.png')
    img_nano = Image.open('images/svm.jpg')

    # Part I
    with st.container():
        st.subheader('Hi, I am rzlAI:tiger:')
        st.title('A Researcher from FJNU')
        st.write(
            "I am passionate on artificial intelligence technology..."
        )
        st.write("[Learn More...](https://space.bilibili.com/32861890?spm_id_from=333.788.0.0)")

    # Part Ⅱ
    with st.container():
        st.write("---")
        l_column, r_column = st.columns(2)
        with l_column:
            st.header("What I Do")
            st.write('###')
            st.write(
                """
                The efficacy evaluation of Port wine stain (PWS) mainly relies on subjective evaluation by doctors, 
                which has drawbacks such as low accuracy and strong subjectivity in the evaluation results. 
                This article aims to build an automated evaluation system for the efficacy of PWS, which can quickly 
                and accurately evaluate the efficacy of PWS and assist doctors in formulating follow-up treatment plans 
                for different patients.This article proposes a therapeutic evaluation system based on the FSASI scoring 
                system. In the image segmentation stage, Retinaface is used to achieve facial segmentation, and neural 
                network image segmentation algorithms such as Deeplabv3+, BiseNet, etc. are used to extract fresh red 
                stains and normal facial skin; In the efficacy evaluation stage, this article creatively proposes a 
                feature vector that conforms to human perception and specifically targets the color features of PWS. 
                This feature vector is based on the HSV color quantization histogram, subdivided into quantization
                 levels, and fully extracts the color information of PWS. Finally, a support vector machine (SVM) model 
                 is constructed for color rating to establish a efficacy evaluation model. The Kappa consistency 
                 analysis results of the experiment indicate that the algorithm proposed in this article has achieved 
                 good fitting results with the subjective observation of doctors, proving the effectiveness of the 
                 high-precision intelligent automatic diagnosis system for the efficacy of PWS.
                """
            )
        with r_column:
            st_lottie(lottie_coding, height=300, key='coding')

    # part Ⅲ
    with st.container():
        # pws
        st.write('---')
        st.header('系统原理')
        st.write('##')
        f_column, l_column, r_column = st.columns(3)
        with f_column:
            image_col, text_col = st.columns((1, 2))
            with image_col:
                st.image(img_sphere)
            with text_col:
                st.write(
                    """
                    鲜红斑痣图片
                    """
                )
                st.write("[Learn More...](https://baike.baidu.com/item/%E9%B2%9C%E7%BA%A2%E6%96%91%E7%97%A3/2227583)")
        with l_column:
            image_col, text_col = st.columns((1, 2))
            with image_col:
                st.image(img_phase_separation)
            with text_col:
                st.write(
                    """
                    FSASI评分模型
                    """
                )
                st.write("[Learn More...](https://pubmed.ncbi.nlm.nih.gov/34741790/)")
        with r_column:
            image_col, text_col = st.columns((1, 2))
            with image_col:
                st.image(img_nano)
            with text_col:
                st.write(
                    """
                    支持向量机
                    """
                )
                st.write("[Learn More...](https://baike.baidu.com/item/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/9683835)")

    with st.container():
        st.write("---")
        st.header("Get In Touch With Me")
        st.write('##')
        st.write(
            """
            2441431330@qq.com
            """
        )

