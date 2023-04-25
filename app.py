# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: app.py
@time: 2023/4/19 23:47
"""

import streamlit as st
from multipage import MultiPage
from Pages import home, machine_learning

st.set_page_config(page_title='MLApp', page_icon=':tiger:', layout='wide')
st.title('鲜红斑痣疗效高精度智能自动诊断系统')


app = MultiPage()

# add application
app.add_page('Home', home.app)
app.add_page('Machine Learning', machine_learning.app)


if __name__ == '__main__':
    app.run()