
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'register.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication
from main.paper_test import RegisterHelper


class Ui_Dialog(object):

    def __init__(self):
        # self.source_imagePath = '../datasets/row_data/multispectral/door1.jpg'
        # self.target_imagePath = '../datasets/row_data/multispectral/door1.jpg'
        self.source_imagePath = None
        self.target_imagePath = None
        self.itermax = 800


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        # Dialog.setStyleSheet("background-color: rgb(255, 251, 246);")
        Dialog.resize(864, 706)
        self.Btn_select_floating = QtWidgets.QPushButton(Dialog)
        self.Btn_select_floating.setGeometry(QtCore.QRect(220, 280, 120, 41))
        self.Btn_select_floating.setObjectName("Btn_select_floating")
        self.Btn_select_reference = QtWidgets.QPushButton(Dialog)
        self.Btn_select_reference.setGeometry(QtCore.QRect(220, 490, 120, 41))
        self.Btn_select_reference.setObjectName("Btn_select_reference")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(90, 370, 101, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(90, 600, 74, 22))
        self.label_2.setObjectName("label_2")
        self.Label_img_floating = QtWidgets.QLabel(Dialog)
        self.Label_img_floating.setGeometry(QtCore.QRect(30, 210, 181, 161))
        self.Label_img_floating.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255));")
        self.Label_img_floating.setText("")
        self.Label_img_floating.setObjectName("Label_img_floating")
        self.Label_img_reference = QtWidgets.QLabel(Dialog)
        self.Label_img_reference.setGeometry(QtCore.QRect(30, 420, 181, 161))
        self.Label_img_reference.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255));")
        self.Label_img_reference.setText("")
        self.Label_img_reference.setObjectName("Label_img_reference")
        self.Label_img_CNN = QtWidgets.QLabel(Dialog)
        self.Label_img_CNN.setGeometry(QtCore.QRect(400, 140, 181, 161))
        self.Label_img_CNN.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255));")
        self.Label_img_CNN.setText("")
        self.Label_img_CNN.setObjectName("Label_img_CNN")
        self.Btn_register_CNN = QtWidgets.QPushButton(Dialog)
        self.Btn_register_CNN.setGeometry(QtCore.QRect(600, 210, 110, 41))
        self.Btn_register_CNN.setObjectName("Btn_register_CNN")
        self.Btn_register_NTG = QtWidgets.QPushButton(Dialog)
        self.Btn_register_NTG.setGeometry(QtCore.QRect(600, 400, 110, 41))
        self.Btn_register_NTG.setObjectName("Btn_register_NTG")
        self.Btn_register_mixed = QtWidgets.QPushButton(Dialog)
        self.Btn_register_mixed.setGeometry(QtCore.QRect(600, 590, 110, 41))
        self.Btn_register_mixed.setObjectName("Btn_register_mixed")
        # self.Txt_maxiter = QtWidgets.QPlainTextEdit(Dialog)
        # self.Txt_maxiter.setGeometry(QtCore.QRect(160, 140, 80, 41))
        # font = QtGui.QFont()
        # font.setPointSize(12)
        # self.Txt_maxiter.setFont(font)
        # self.Txt_maxiter.setLayoutDirection(QtCore.Qt.LeftToRight)
        # self.Txt_maxiter.setObjectName("Txt_maxiter")

        self.Label_img_NTG = QtWidgets.QLabel(Dialog)
        self.Label_img_NTG.setGeometry(QtCore.QRect(400, 330, 181, 161))
        self.Label_img_NTG.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255));")
        self.Label_img_NTG.setText("")
        self.Label_img_NTG.setObjectName("Label_img_NTG")
        self.Label_img_mixed = QtWidgets.QLabel(Dialog)
        self.Label_img_mixed.setGeometry(QtCore.QRect(400, 520, 181, 161))
        self.Label_img_mixed.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255));")
        self.Label_img_mixed.setText("")
        self.Label_img_mixed.setObjectName("Label_img_mixed")
        # self.Btn_setIterMax = QtWidgets.QPushButton(Dialog)
        # self.Btn_setIterMax.setGeometry(QtCore.QRect(60, 140, 101, 41))
        # self.Btn_setIterMax.setObjectName("Btn_setIterMax")
        self.Label_tText_status = QtWidgets.QLabel(Dialog)
        self.Label_tText_status.setGeometry(QtCore.QRect(30, 660, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Label_tText_status.setFont(font)
        self.Label_tText_status.setObjectName("Label_tText_status")

        self.label_3 = QtWidgets.QLabel(Dialog)
        # self.label_3.setGeometry(QtCore.QRect(0, 0, 871, 71))
        self.label_3.setGeometry(QtCore.QRect(0, 30, 871, 71))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        # self.label_3.setStyleSheet("background-color: rgb(255, 251, 246);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.Time_cnn_label = QtWidgets.QLabel(Dialog)
        self.Time_cnn_label.setGeometry(QtCore.QRect(720, 210, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Time_cnn_label.setFont(font)
        self.Time_cnn_label.setObjectName("Time_cnn_label")
        self.time_ntg_label = QtWidgets.QLabel(Dialog)
        self.time_ntg_label.setGeometry(QtCore.QRect(720, 400, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.time_ntg_label.setFont(font)
        self.time_ntg_label.setObjectName("time_ntg_label")
        self.time_mix_label = QtWidgets.QLabel(Dialog)
        self.time_mix_label.setGeometry(QtCore.QRect(720, 590, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.time_mix_label.setFont(font)
        self.time_mix_label.setObjectName("time_mix_label")

        self.Btn_register_CNN.clicked.connect(self.registerCNN)
        self.Btn_register_NTG.clicked.connect(self.registerNTG)
        self.Btn_register_mixed.clicked.connect(self.registerMixed)
        self.Btn_select_floating.clicked.connect(self.openImage1)
        self.Btn_select_reference.clicked.connect(self.openImage2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "多光谱图像配准系统"))
        self.Btn_select_floating.setText(_translate("Dialog", "选择待配准图片"))
        self.Btn_select_reference.setText(_translate("Dialog", "选择模板图片"))
        self.label.setText(_translate("Dialog", "待配准图片"))
        self.label_2.setText(_translate("Dialog", "模板图片"))
        self.Btn_register_CNN.setText(_translate("Dialog", "CNN模型配准"))
        self.Btn_register_NTG.setText(_translate("Dialog", "NTG迭代配准"))
        self.Btn_register_mixed.setText(_translate("Dialog", "混合框架配准"))
        # self.Txt_maxiter.setPlainText(_translate("Dialog", "800"))
        # self.Btn_setIterMax.setText(_translate("Dialog", "设置迭代次数"))
        self.Label_tText_status.setText(_translate("Dialog", "正在初始化..."))
        self.label_3.setText(_translate("Dialog", "多光谱图像配准系统"))
        self.registerHelper = RegisterHelper()
        self.Label_tText_status.setText("初始化完成")


    def openImage1(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.Label_img_floating.width(), self.Label_img_floating.height())
        self.source_imagePath = imgName
        self.Label_img_floating.setPixmap(jpg)

    def openImage2(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.Label_img_reference.width(), self.Label_img_reference.height())
        self.target_imagePath = imgName
        self.Label_img_reference.setPixmap(jpg)

    def registerCNN(self):
        self.register('cnn')

    def registerNTG(self):
        self.register('ntg')

    def registerMixed(self):
        self.register('mixed')


    def register(self,type='cnn'):
        '''
        :param type: cnn  |  ntg | cnn_ntg
        :return:
        '''

        # self.itermax = int(self.Txt_maxiter.toPlainText())
        # print(self.source_imagePath,self.target_imagePath," type:"+type,"itermax:"+str(self.itermax))

        QApplication.processEvents()

        time_start = time.time()
        if type == 'cnn':
            self.Label_tText_status.setText('使用CNN配准...')
            image = self.registerHelper.register_CNN(self.source_imagePath,self.target_imagePath)
            time_end = time.time()
        elif type == 'ntg':
            self.Label_tText_status.setText('使用NTG配准...')
            self.itermax = 800
            image = self.registerHelper.register_NTG(self.source_imagePath,self.target_imagePath,itermax=self.itermax)
            time_end = time.time()
        else:
            self.itermax = 30
            self.Label_tText_status.setText('使用混合框架配准...')
            image = self.registerHelper.register_CNN_NTG(self.source_imagePath,self.target_imagePath,itermax=self.itermax,custom_pyramid_level=2)
            time_end = time.time()
        print("配准完成，开始显示图片")
        self.Label_tText_status.setText('配准完成，开始显示图片')
        image = QtGui.QImage(image,image.shape[1],image.shape[0],QtGui.QImage.Format_RGB888)

        if type == 'cnn':
            pix = QtGui.QPixmap(image).scaled(self.Label_img_CNN.width(), self.Label_img_CNN.height())
            time_cost = time_end-time_start
            self.Label_img_CNN.setPixmap(pix)
            self.Label_img_CNN.update()
            self.Time_cnn_label.setText("耗时："+str(round(time_cost,2))+"秒")
        elif type == 'ntg':
            pix = QtGui.QPixmap(image).scaled(self.Label_img_NTG.width(), self.Label_img_NTG.height())
            time_cost = time_end - time_start
            self.Label_img_NTG.setPixmap(pix)
            self.Label_img_NTG.update()
            self.time_ntg_label.setText("耗时：" + str(round(time_cost, 2)) + "秒")
        else:
            pix = QtGui.QPixmap(image).scaled(self.Label_img_mixed.width(), self.Label_img_mixed.height())
            time_cost = time_end - time_start
            self.Label_img_mixed.setPixmap(pix)
            self.Label_img_mixed.update()
            self.time_mix_label.setText("耗时：" + str(round(time_cost, 2)) + "秒")

        # print("显示完成")
        # print("耗时：",str(time_cost))
