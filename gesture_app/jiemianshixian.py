# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:59:14 2022

@author: admin
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import uic
from .interaction_6 import Ui_Form
from PyQt5.QtGui import QFont
from functools import partial


# 动态载入
class mainwindow(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        # PyQt5
        # 使用包内相对路径加载 UI
        from . import paths
        self.ui=uic.loadUi(str(paths.ui_dir() / "interaction_6.ui"))
        self.ui.letter_rec.clicked.connect(self.letter_rec)
        self.ui.number_rec.clicked.connect(self.number_rec)
        self.ui.textBrowser.setFont(QFont('华文楷体',20))
        self.ui.rec_again.clicked.connect(self.clear_log)
        self.ui.number_gesture.clicked.connect(self.number_gesture)
        self.ui.letter_gesture.clicked.connect(self.letter_gesture)
        self.ui.number_rec_2.clicked.connect(self.dynamic_rec)
        # 指向方向识别（上/下/左/右）
        if hasattr(self.ui, 'direction_gesture'):
            self.ui.direction_gesture.clicked.connect(self.direction_gesture)
        # 新增：动态字母手势 J/Z
        if hasattr(self.ui, 'letter_gesture_dynamic'):
            self.ui.letter_gesture_dynamic.clicked.connect(self.dynamic_letter_gesture)
        # self.ui.written_letter.clicked.connect(self.written_letter)
        # self.ui.written_number.clicked.connect(self.written_number)
        # self.ui.speech.clicked.connect(self.speech)
        
        #self.ui.
        # 这里与静态载入不同，使用 self.ui.show()
        # 如果使用 self.show(),会产生一个空白的 MainWindow
        #self.ui.show()
    
    def letter_rec(self):
        from .letter_rec import letter_rec
        letter_rec(self.ui)
        
    def number_rec(self):
        from .number_rec import number_rec
        number_rec(self.ui)
        
    def number_gesture(self):
        from .number_gesture import number_gesture
        number_gesture(self.ui)
        
    def letter_gesture(self):
        from .letter_gesture import letter_gesture
        letter_gesture(self.ui)

    def dynamic_letter_gesture(self):
        from .letter_gesture import JZRec
        JZRec(self.ui)
        
    def dynamic_rec(self):
        from .dg_prediction_CSRN import CSRN
        CSRN(self.ui)
        
    def clear_log(self):
        self.ui.textBrowser.clear()
    
    def direction_gesture(self):
        # 集成方向识别模块（已迁入包内）
        from .direction_gesture import direction_gesture
        direction_gesture(self.ui)
    # def try1(self):
    #     if self.ui.letter_rec.isDown():
    #         letter_rec()


if __name__=="__main__":
    app=QApplication(sys.argv)
    window=mainwindow()
    window.ui.show()
    sys.exit(app.exec_())

# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
# from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
# from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
# from PyQt5.QtCore import Qt, QPoint, QSize,QFile
# from PyQt5.QtUiTools import QUiLoader

# from interaction import Ui_Form


# import sys, os
# import numpy as np
# from PIL import Image, ImageQt

# class interaction:

#     def __init__(self):
#         # 从文件中加载UI定义
#         qfile_inter = QFile('interaction.ui')
#         qfile_inter.open(QFile.ReadOnly)
#         qfile_inter.close()
        
#         # 从 UI 定义中动态 创建一个相应的窗口对象
#         # 注意：里面的控件对象也成为窗口对象的属性了
#         # 比如 self.ui.button , self.ui.textEdit
#         self.ui = QUiLoader().load(qfile_inter)

#         self.ui.recagain.clicked.connect(self.try1)
        
        
#     def try1(self):
#         if self.recagain.isChecked():
#             print('1')
            
# if __name__=='__main__':
#     app = QApplication(sys.argv)
#     interaction = interaction()
#     interaction.show()
#     sys.exit(app.exec_())
    
        
