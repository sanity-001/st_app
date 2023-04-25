# -*- coding:utf-8 -*-

"""
@author: 阮智霖
@software: Pycharm
@file: multipage.py
@time: 2023/4/19 23:58
"""

import streamlit as st


class MultiPage:
    def __init__(self) -> None:
        self.pages = []

    def add_page(self, title, func):
        self.pages.append(
            {
                'title': title,
                'function': func
            }
        )

    def run(self):
        page = st.sidebar.selectbox(
            'App navigation',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()
