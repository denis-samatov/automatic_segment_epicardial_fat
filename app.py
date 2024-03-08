from PyQt5.QtWidgets import QLabel, QWidget, QPushButton, QFileDialog, QMessageBox, QVBoxLayout, QMainWindow, QTableWidgetItem, QTableWidget, QDesktopWidget
from PyQt5.QtGui import QBrush, QPen, QPainter, QImage, QPixmap, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd
import sys
import os
import cv2 as cv
import numpy as np
import shutil


import _automatic_
import _semiautomatic_
import resources

from algorithm.Show3DModel import plt_plot_3d

global flag


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        'STYLESHEET'
        buttonStyle = """
        QPushButton{
                border-style: solid;
                border-width: 0px;
                border-radius: 10px;
                background-color: rgb(234,234,234);
        }
        QPushButton::hover{
                background-color: rgb(183,181,181);
        }
        """
        groupBoxStyle = """
        QGroupBox{
                background-color:none;
                border: 1px solid rgb(183,181,181);
                border-radius: 10px;
        }
        """

        closeStyle = """
        QPushButton{
            border-style: solid;
            border-width: 0px;
            border-radius: 10px;
            background-color: rgb(234,67,53);
            color:rgb(255, 255, 255)
        }
        QPushButton::hover{
            background-color: rgb(172,16,12);
        }
        """

        generateVolumeStyle = """
        QPushButton{
                border-style: solid;
                border-width: 0px;
                border-radius: 10px;
                background-color: rgb(70,136,244);
                color:rgb(255, 255, 255)
        }

        QPushButton::hover
        {
                background-color: rgb(18,77,150);
        }
        """

        boxStyle = """
        QGroupBox{
            border-style: none;
            background-color:none;
        }
        """

        textEditStyle = """
        QTextEdit{
            border-style: solid;
            border-width: 1px;
            border-radius: 10px;
            border-color: rgb(183,181,181);
        }
        """

        lineEditStyle = """
        QLineEdit{
            border-style: solid;
            border-width: 1px;
            border-radius: 10px;
            border-color: rgb(183,181,181);
        }
        """

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1208, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1208, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1208, 700))
        MainWindow.setMouseTracking(False)
        MainWindow.setTabletTracking(False)
        MainWindow.setStyleSheet("background-color:rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.filename = QtWidgets.QLineEdit(self.centralwidget)
        self.filename.setGeometry(QtCore.QRect(740, 50, 411, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.filename.setFont(font)
        self.filename.setStyleSheet(lineEditStyle)
        self.filename.setFrame(False)
        self.filename.setObjectName("filename")
        self.openFile = QtWidgets.QPushButton(self.centralwidget)
        self.openFile.setGeometry(QtCore.QRect(600, 50, 131, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.openFile.setFont(font)
        self.openFile.setStyleSheet(buttonStyle)
        self.openFile.setObjectName("openFile")
        self.imageHeart = QtWidgets.QLabel(self.centralwidget)
        self.imageHeart.setEnabled(True)
        self.imageHeart.setGeometry(QtCore.QRect(30, 20, 520, 520))
        self.imageHeart.setText("")
        self.imageHeart.setPixmap(QtGui.QPixmap(":/aux/default_txt.png"))
        self.imageHeart.setScaledContents(True)
        self.imageHeart.setAlignment(QtCore.Qt.AlignCenter)
        self.imageHeart.setObjectName("imageHeart")
        self.manual_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.manual_intervention.setEnabled(True)
        self.manual_intervention.setGeometry(QtCore.QRect(590, 320, 591, 211))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        self.manual_intervention.setFont(font)
        self.manual_intervention.setStyleSheet(groupBoxStyle)
        self.manual_intervention.setTitle("")
        self.manual_intervention.setObjectName("manual_intervention")
        self.duringEditionBox = QtWidgets.QGroupBox(self.manual_intervention)
        self.duringEditionBox.setGeometry(QtCore.QRect(0, 15, 571, 91))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.duringEditionBox.setFont(font)
        self.duringEditionBox.setStyleSheet(boxStyle)
        self.duringEditionBox.setTitle("")
        self.duringEditionBox.setObjectName("duringEditionBox")
        self.generateVolumeSemi = QtWidgets.QPushButton(self.duringEditionBox)
        self.generateVolumeSemi.setGeometry(QtCore.QRect(430, 0, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.generateVolumeSemi.setFont(font)
        self.generateVolumeSemi.setMouseTracking(False)
        self.generateVolumeSemi.setStyleSheet(generateVolumeStyle)
        self.generateVolumeSemi.setObjectName("generateVolumeSemi")
        self.slices = QtWidgets.QTextEdit(self.duringEditionBox)
        self.slices.setGeometry(QtCore.QRect(130, 3, 281, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.slices.setFont(font)
        self.slices.setStyleSheet(textEditStyle)
        self.slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slices.setReadOnly(True)
        self.slices.setObjectName("slices")
        self.slices_edited = QtWidgets.QLabel(self.duringEditionBox)
        self.slices_edited.setGeometry(QtCore.QRect(30, 10, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.slices_edited.setFont(font)
        self.slices_edited.setObjectName("slices_edited")
        self.infoAuto_2 = QtWidgets.QGroupBox(self.duringEditionBox)
        self.infoAuto_2.setGeometry(QtCore.QRect(320, 50, 251, 41))
        self.infoAuto_2.setStyleSheet(boxStyle)
        self.infoAuto_2.setTitle("")
        self.infoAuto_2.setObjectName("infoAuto_2")
        self.textAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.textAuto_2.setGeometry(QtCore.QRect(30, 10, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(7)
        font.setBold(False)
        font.setWeight(50)
        self.textAuto_2.setFont(font)
        self.textAuto_2.setStyleSheet("background-color:none;")
        self.textAuto_2.setObjectName("textAuto_2")
        self.iconAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.iconAuto_2.setGeometry(QtCore.QRect(10, 10, 15, 15))
        self.iconAuto_2.setAutoFillBackground(False)
        self.iconAuto_2.setText("")
        self.iconAuto_2.setPixmap(QtGui.QPixmap(r'.\resources\info.png'))
        self.iconAuto_2.setScaledContents(True)
        self.iconAuto_2.setObjectName("iconAuto_2")
        self.afterEditionBox = QtWidgets.QGroupBox(self.manual_intervention)
        self.afterEditionBox.setGeometry(QtCore.QRect(0, 100, 571, 101))
        self.afterEditionBox.setStyleSheet(boxStyle)
        self.afterEditionBox.setTitle("")
        self.afterEditionBox.setObjectName("afterEditionBox")
        self.volume_2 = QtWidgets.QTextEdit(self.afterEditionBox)
        self.volume_2.setGeometry(QtCore.QRect(30, 45, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.volume_2.setFont(font)
        self.volume_2.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.volume_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.volume_2.setReadOnly(True)
        self.volume_2.setObjectName("volume_2")
        self.new_volume = QtWidgets.QLabel(self.afterEditionBox)
        self.new_volume.setGeometry(QtCore.QRect(30, 20, 180, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.new_volume.setFont(font)
        self.new_volume.setObjectName("new_volume")
        self.radiomicsButtonSemi = QtWidgets.QPushButton(self.afterEditionBox)
        self.radiomicsButtonSemi.setGeometry(QtCore.QRect(430, 10, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radiomicsButtonSemi.setFont(font)
        self.radiomicsButtonSemi.setStyleSheet(buttonStyle)
        self.radiomicsButtonSemi.setObjectName("countRadiomics")
        self.view3DSemi = QtWidgets.QPushButton(self.afterEditionBox)
        self.view3DSemi.setGeometry(QtCore.QRect(430, 70, 131, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.view3DSemi.setFont(font)
        self.view3DSemi.setStyleSheet(buttonStyle)
        self.view3DSemi.setObjectName("view3DSemi")
        self.dicom_file = QtWidgets.QLabel(self.centralwidget)
        self.dicom_file.setGeometry(QtCore.QRect(600, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.dicom_file.setFont(font)
        self.dicom_file.setObjectName("dicom_file")
        self.generateVolumeAuto = QtWidgets.QPushButton(self.centralwidget)
        self.generateVolumeAuto.setEnabled(True)
        self.generateVolumeAuto.setGeometry(QtCore.QRect(1020, 90, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.generateVolumeAuto.setFont(font)
        self.generateVolumeAuto.setMouseTracking(False)
        self.generateVolumeAuto.setStyleSheet(generateVolumeStyle)
        self.generateVolumeAuto.setObjectName("generateVolumeAuto")
        self.automatic_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.automatic_intervention.setEnabled(True)
        self.automatic_intervention.setGeometry(QtCore.QRect(590, 180, 591, 111))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.automatic_intervention.setFont(font)
        self.automatic_intervention.setStyleSheet(groupBoxStyle)
        self.automatic_intervention.setTitle("")
        self.automatic_intervention.setObjectName("automatic_intervention")
        self.radiomicsButton = QtWidgets.QPushButton(self.automatic_intervention)
        self.radiomicsButton.setGeometry(QtCore.QRect(430, 10, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radiomicsButton.setFont(font)
        self.radiomicsButton.setStyleSheet(buttonStyle)
        self.radiomicsButton.setObjectName("countRadiomics")
        self.view3DAuto = QtWidgets.QPushButton(self.automatic_intervention)
        self.view3DAuto.setGeometry(QtCore.QRect(430, 70, 131, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.view3DAuto.setFont(font)
        self.view3DAuto.setStyleSheet(buttonStyle)
        self.view3DAuto.setObjectName("view3DAuto")
        self.volume_detected = QtWidgets.QLabel(self.automatic_intervention)
        self.volume_detected.setGeometry(QtCore.QRect(30, 30, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.volume_detected.setFont(font)
        self.volume_detected.setObjectName("volume_detected")
        self.volume = QtWidgets.QLineEdit(self.automatic_intervention)
        self.volume.setGeometry(QtCore.QRect(30, 55, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.volume.setFont(font)
        self.volume.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.volume.setText("")
        self.volume.setFrame(False)
        self.volume.setReadOnly(True)
        self.volume.setObjectName("volume")
        self.slicesManager = QtWidgets.QGroupBox(self.centralwidget)
        self.slicesManager.setEnabled(False)
        self.slicesManager.setGeometry(QtCore.QRect(30, 550, 521, 101))
        self.slicesManager.setStyleSheet(boxStyle)
        self.slicesManager.setTitle("")
        self.slicesManager.setObjectName("slicesManager")
        self.total_slices = QtWidgets.QTextEdit(self.slicesManager)
        self.total_slices.setGeometry(QtCore.QRect(300, 10, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.total_slices.setFont(font)
        self.total_slices.setMouseTracking(True)
        self.total_slices.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.total_slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.total_slices.setLineWidth(1)
        self.total_slices.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.total_slices.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.total_slices.setPlaceholderText("")
        self.total_slices.setObjectName("total_slices")
        self.label_2 = QtWidgets.QLabel(self.slicesManager)
        self.label_2.setGeometry(QtCore.QRect(180, 10, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.of = QtWidgets.QLabel(self.slicesManager)
        self.of.setGeometry(QtCore.QRect(260, 10, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.of.setFont(font)
        self.of.setAlignment(QtCore.Qt.AlignCenter)
        self.of.setObjectName("of")
        self.previous = QtWidgets.QPushButton(self.slicesManager)
        self.previous.setGeometry(QtCore.QRect(10, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.previous.setFont(font)
        self.previous.setStyleSheet(buttonStyle)
        self.previous.setObjectName("previous")
        self.next = QtWidgets.QPushButton(self.slicesManager)
        self.next.setGeometry(QtCore.QRect(420, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.next.setFont(font)
        self.next.setStyleSheet(buttonStyle)
        self.next.setObjectName("next")
        self.edit_slice = QtWidgets.QPushButton(self.slicesManager)
        self.edit_slice.setGeometry(QtCore.QRect(180, 60, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.edit_slice.setFont(font)
        self.edit_slice.setStyleSheet(buttonStyle)
        self.edit_slice.setObjectName("edit_slice")
        self.actual_slice = QtWidgets.QTextEdit(self.slicesManager)
        self.actual_slice.setGeometry(QtCore.QRect(232, 10, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.actual_slice.setFont(font)
        self.actual_slice.setMouseTracking(True)
        self.actual_slice.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.actual_slice.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.actual_slice.setLineWidth(1)
        self.actual_slice.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.actual_slice.setReadOnly(True)
        self.actual_slice.setPlaceholderText("")
        self.actual_slice.setObjectName("actual_slice")
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setGeometry(QtCore.QRect(1020, 600, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.closeButton.setFont(font)
        self.closeButton.setStyleSheet(closeStyle)
        self.closeButton.setObjectName("closeButton")
        self.autoLabel = QtWidgets.QLabel(self.centralwidget)
        self.autoLabel.setGeometry(QtCore.QRect(610, 170, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.autoLabel.setFont(font)
        self.autoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.autoLabel.setObjectName("autoLabel")
        self.semiLabel = QtWidgets.QLabel(self.centralwidget)
        self.semiLabel.setGeometry(QtCore.QRect(610, 310, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.semiLabel.setFont(font)
        self.semiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.semiLabel.setObjectName("semiLabel")
        self.infoAuto = QtWidgets.QGroupBox(self.centralwidget)
        self.infoAuto.setGeometry(QtCore.QRect(910, 140, 251, 41))
        self.infoAuto.setStyleSheet(boxStyle)
        self.infoAuto.setTitle("")
        self.infoAuto.setObjectName("infoAuto")
        self.textAuto = QtWidgets.QLabel(self.infoAuto)
        self.textAuto.setGeometry(QtCore.QRect(30, 8, 221, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(7)
        font.setBold(False)
        font.setWeight(50)
        self.textAuto.setFont(font)
        self.textAuto.setStyleSheet("background-color:none;")
        self.textAuto.setObjectName("textAuto")
        self.iconAuto = QtWidgets.QLabel(self.infoAuto)
        self.iconAuto.setGeometry(QtCore.QRect(10, 10, 15, 15))
        self.iconAuto.setAutoFillBackground(False)
        self.iconAuto.setText("")
        self.iconAuto.setPixmap(QtGui.QPixmap(r'.\resources\info.png'))
        self.iconAuto.setScaledContents(True)
        self.iconAuto.setObjectName("iconAuto")
        self.automatic_intervention.raise_()
        self.manual_intervention.raise_()
        self.filename.raise_()
        self.openFile.raise_()
        self.imageHeart.raise_()
        self.dicom_file.raise_()
        self.generateVolumeAuto.raise_()
        self.slicesManager.raise_()
        self.closeButton.raise_()
        self.autoLabel.raise_()
        self.semiLabel.raise_()
        self.infoAuto.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1208, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")

        'Flag to manager the text shown in slices label'
        self.newEdition = False

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    'WINDOWS MANAGER'
    def searchFolder(self):
        global dir
        dir = QFileDialog.getExistingDirectory()

        # Set the font for the placeholder text
        font = QFont("Roboto")
        self.filename.setFont(font)

        self.filename.setPlaceholderText("Specify the path to the folder with DICOM images")
        self.filename.setText(dir)
        self.enableGenerateVolume()

    def openSecondWindow(self):
        self.clearEditLine()

        'Variables to send to edit window'
        self.patient_edit = patient_id
        current_slice = int(self.actual_slice.toPlainText())
        self.slice_edit = current_slice

        "Send data to second window"
        self.secondwindow = SecondWindow(self.slice_edit, self.patient_edit)
        self.secondwindow.submited.connect(self.updateImage)
    
    def openThirdWindow(self, volume_type):
        self.clearEditLine()

        self.thirdwindow = ThirdWindow(volume_type)

    def openFourthWindow(self):
        self.clearEditLine()

        self.fourthWindow = FourthWindow()

    def closeProgram(self):
        self.cleanAll()
        MainWindow.close()

    def cleanAll(self):
        result_folder_path = r'result_folder'

        # Create a custom QMessageBox
        msgBox = QMessageBox(MainWindow)

        # Set the font for the QMessageBox
        font = QFont("Roboto")
        font.setPointSize(11)
        msgBox.setFont(font)

        msgBox.setWindowTitle('Delete Segmentation Results')
        msgBox.setText('Do you want to delete the segmentation results?')


        # Create custom buttons
        button_width = 50
        button_height = 30

        # Create custom buttons with your styles and set the geometry
        yes_button = msgBox.addButton('Yes', QMessageBox.YesRole)
        yes_button.setFixedSize(button_width, button_height)

        no_button = msgBox.addButton('No', QMessageBox.NoRole)
        no_button.setFixedSize(button_width, button_height)

        # Set the style sheet for the buttons (customize as needed)
        font = QtGui.QFont()
        font.setPointSize(12)
        button_style = """
            QPushButton {{
                border: 2px solid {border_color};
                border-radius: 10px;
                background-color: {background_color};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-color: {border_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {pressed_border_color};
            }}
        """

        # Apply styles to buttons
        yes_style = button_style.format(
            border_color="rgb(70, 136, 244)",
            background_color="rgb(70, 136, 244)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(18,77,150)",
            pressed_color="rgb(80, 206, 209)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        yes_button.setStyleSheet(yes_style)

        no_style = button_style.format(
            border_color="rgb(224, 9, 22)",
            background_color="rgb(224, 9, 22)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(172, 16, 12)",
            pressed_color="rgb(224, 54, 80)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        no_button.setStyleSheet(no_style)

        # Set the default button to "No"
        msgBox.setDefaultButton(no_button)

        # Show the custom QMessageBox and handle the user's choice
        clicked_button = msgBox.exec_()
        if msgBox.clickedButton() == yes_button:
            # Delete the result folder and its contents
            try:
                shutil.rmtree(result_folder_path)
            except OSError as e:
                QMessageBox.critical(MainWindow, 'Error', f'Error deleting results: {str(e)}')

    def closeEvent(self, event):
        # Ask the user before closing the program
        msgBox = QMessageBox()
        msgBox.setText("Close Program")
        msgBox.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.Yes)

        if msgBox.exec_() == QMessageBox.Yes:
            self.cleanAll()  # Clean only if the user chooses to close
            event.accept()
        else:
            event.ignore()

    'GENERATE AUTOMATIC AND SEMIAUTOMATIC VOLUME'
    def generateVolume(self):
        global patient_id
        global no_slices
        global patient
        patient_id, patient, no_slices, vol = _automatic_.segment_epicardial_fat(DICOM_DATASET=dir, OUTPUT_FOLDER='result_folder/')
        self.volume.setText(str(round(vol, 1)))
        self.refreshAfterAutomatic()

    def generateNewVolume(self):
        patient_id, patient, no_slices, vol = _semiautomatic_.segment_epicardial_fat(DICOM_DATASET=dir, OUTPUT_FOLDER='result_folder/')
        self.volume_2.setText(str(round(vol, 1)))
        self.refreshAfterSemiAutomatic()
    
    'Show 3D FLAT VOLUME'
    # def generate3DFatVolume(self):
    #     filePath = f'result_folder\\{patient}\\fat'
    #     plt_plot_3d(filePath, save_path=f'result_folder/{patient}/3d_fat.png')

    
    # def generateNew3DFatVolume(self):
    #     filePath = f'result_folder\\{patient}\\fat'
    #     plt_plot_3d(filePath, save_path=f'result_folder/{patient}/new_3d_fat.png')

    'Show 3D FLAT VOLUME'
    # def show3DFatVolume(self):
    #     filePath = f'result_folder\\{patient}\\fat'
    #     savePath = f'result_folder/{patient}/3d_fat.png'
    #     ThirdWindow(filePath, savePath)
    
    # def showNew3DFatVolume(self):
    #     filePath = f'result_folder\\{patient}\\fat'
    #     savePath = f'result_folder/{patient}/new_3d_fat.png'
    #     ThirdWindow(filePath, savePath)

    'INTERFACE MANAGER'
    def updateImage(self, slice_id):
        "Show container"
        self.manual_intervention.show()
        self.semiLabel.show()
        self.radiomicsButtonSemi.show()
        self.view3DSemi.show()
        self.afterEditionBox.hide()
        self.slice_id = slice_id
        self.editionHandler()

        'Show the image with new contour'
        self.imageHeart.setPixmap(QtGui.QPixmap(':/aux/edited_successfully_bg.png'))

    def clearEditLine(self):
        'Clean the slices label'
        if self.newEdition:
            self.slices.setText('')
            self.newEdition = False

    def refreshAfterAutomatic(self):
        self.infoAuto.hide()
        self.automatic_intervention.show()
        self.autoLabel.show()
        self.slicesManager.setEnabled(True)
        self.setFirstSlice()

    def refreshAfterSemiAutomatic(self):
        self.infoAuto_2.hide()
        self.afterEditionBox.show()
        self.setFirstSlice()
        self.newEdition = True

    def enableGenerateVolume(self):
        if self.filename.text():
            self.generateVolumeAuto.show()
            self.infoAuto.show()
            self.radiomicsButton.show()
            self.view3DAuto.show()

    'SLICE NAVIGATOR'
    def setFirstSlice(self):
        'Show the number of slices and the first slice'
        self.total_slices.setText(str(no_slices - 1))
        self.actual_slice.setText(str(1))
        'Show image of slice 0'
        self.imageHeart.setPixmap(QtGui.QPixmap(f"result_folder/{patient}/combined/{patient_id}_{0}.png"))

    def getCurrentSlice(self):
        return self.actual_slice.toPlainText()

    def displaySlice(self):
        current = self.getCurrentSlice()
        slice = int(current) - 1
        self.imageHeart.setPixmap(QtGui.QPixmap(f"result_folder/{patient}/combined/{patient_id}_{slice}.png"))

    def nextSlice(self):
        current = int(self.getCurrentSlice())
        if current < no_slices - 1:
            self.actual_slice.setText(str(current + 1))
        else:
            self.actual_slice.setText(str(no_slices - 1))
        self.displaySlice()

    def previousSlice(self):
        current = int(self.getCurrentSlice())
        if current > 1:
            self.actual_slice.setText(str(current - 1))
        else:
            self.actual_slice.setText(str(1))
        self.displaySlice()

    def editionHandler(self):
        current = int(self.getCurrentSlice())
        current_text = self.slices.toPlainText()

        if not current_text:
            self.slices.setText(str(current))
        else:
            current_text = f'{current_text}, {current}'
            self.slices.setText(str(current_text))

    'RETRANSLATE UI'
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Epicardial Fat Segmentation Software", "Epicardial Fat Segmentation Software"))
        MainWindow.setWindowIcon(QtGui.QIcon(r".\resources\logo_w.png"))
        MainWindow.setIconSize(QtCore.QSize(500, 200))
        self.openFile.setText(_translate("MainWindow", "Open folder"))
        self.generateVolumeSemi.setText(_translate("MainWindow", "Calculate New \n"
                                                                 "Volume"))
        self.slices_edited.setText(_translate("MainWindow", "Slices edited:"))
        self.textAuto_2.setText(_translate("MainWindow", "This process may takes a few seconds."))
        self.new_volume.setText(_translate("MainWindow", "Epicardial Fat Volume (ml)"))
        self.radiomicsButtonSemi.setText(_translate("MainWindow", "Calculate \n"
                                                                 "Radiomics Features"))
        self.view3DSemi.setText(_translate("MainWindow", "3D View"))
        self.dicom_file.setText(_translate("MainWindow", "DICOM File"))
        self.generateVolumeAuto.setText(_translate("MainWindow", "Calculate \n"
                                                                 "Volume"))
                                                                 
        self.radiomicsButton.setText(_translate("MainWindow", "Calculate \n"
                                                                 "Radiomics Features"))
        self.view3DAuto.setText(_translate("MainWindow", "3D View"))
        self.volume_detected.setText(_translate("MainWindow", "Epicardial Fat Volume (ml)"))
        self.label_2.setText(_translate("MainWindow", "Slice"))
        self.of.setText(_translate("MainWindow", "of"))
        self.previous.setText(_translate("MainWindow", "« Previous"))
        self.next.setText(_translate("MainWindow", "Next »"))
        self.edit_slice.setText(_translate("MainWindow", "Edit slice"))
        self.closeButton.setText(_translate("MainWindow", "Close"))
        self.autoLabel.setText(_translate("MainWindow", "Automatic detection"))
        self.semiLabel.setText(_translate("MainWindow", "Manual intervention"))
        self.textAuto.setText(_translate("MainWindow", "This process may takes a few seconds."))
        self.actionOpen_File.setText(_translate("MainWindow", "Open file"))

        'ASSIGN FUNCTIONS'
        'Hide all elements'
        self.generateVolumeAuto.hide()
        self.automatic_intervention.hide()
        self.manual_intervention.hide()
        self.infoAuto.hide()
        self.autoLabel.hide()
        self.semiLabel.hide()
        
        self.radiomicsButton.hide()
        self.radiomicsButtonSemi.hide()
        self.view3DAuto.hide()
        self.view3DSemi.hide()

        'Button to search file'
        self.openFile.clicked.connect(self.searchFolder)

        'Button to generate volume'
        self.generateVolumeAuto.clicked.connect(self.generateVolume)
        self.generateVolumeSemi.clicked.connect(self.generateNewVolume)

        'Button to generate 3D volume'
        self.view3DAuto.clicked.connect(lambda: self.openThirdWindow("3d_fat"))
        self.view3DSemi.clicked.connect(lambda: self.openThirdWindow("new_3d_fat"))
        # self.view3DAuto.clicked.connect(self.show3DFatVolume)
        # self.view3DSemi.clicked.connect(self.showNew3DFatVolume)

        'Button to count radiomics features'
        self.radiomicsButton.clicked.connect(self.openFourthWindow)
        self.radiomicsButtonSemi.clicked.connect(self.openFourthWindow)

        'Edit slice button'
        self.edit_slice.clicked.connect(self.openSecondWindow)
        
        'Button to go to next and previous slice'
        self.next.clicked.connect(self.nextSlice)
        self.previous.clicked.connect(self.previousSlice)

        'Button to close the program'
        self.closeButton.clicked.connect(self.closeProgram)


'***************************************************************************************************************************************************'
'SECOND WINDOW'
'***************************************************************************************************************************************************'


class SecondWindow(QWidget):
    submited = QtCore.pyqtSignal(int)

    def __init__(self, slice_edit, patient_edit):
        super().__init__()
        self.slice_edit = slice_edit
        self.patient_edit = patient_edit
        self.eventhandler = EventHandler(self)
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Manual Pericardium delineation")
        self.setStyleSheet("background-color:rgb(255, 255, 255);")
        self.setWindowIcon(QtGui.QIcon(r'.\resources\logo_w.png'))

        self.img = self.eventhandler
        img = cv.imread(f"result_folder/{patient}/slices/{self.patient_edit}_{self.slice_edit}.png")
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.img.setPixmap(pixmap)

        'Size of window'
        self.resize(860, 920)

        'Image properties'
        self.img.setMinimumSize(QtCore.QSize(800, 800))
        self.img.setBaseSize(QtCore.QSize(512, 512))
        self.img.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.img.setScaledContents(True)
        self.img.setGeometry(QRect(30, 30, width, height))
        self.img.setCursor(Qt.CrossCursor)

        # Создание стилей для кнопок
        font = QtGui.QFont()
        font.setPointSize(9)
        button_style = """
            QPushButton {{
                border: 2px solid {border_color};
                border-radius: 10px;
                background-color: {background_color};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-color: {border_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {pressed_border_color};
            }}
        """

        self.doneButton = QPushButton('Done', self)
        self.doneButton.setGeometry(QRect(624, 860, 93, 28))
        self.doneButton.setFont(font)
        done_style = button_style.format(
            border_color="rgb(70, 136, 244)",
            background_color="rgb(70, 136, 244)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(18,77,150)",
            pressed_color="rgb(80, 206, 209)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        self.doneButton.setStyleSheet(done_style)
        self.doneButton.clicked.connect(self.onSubmit)


        # Установка свойств кнопки "Cancel"
        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.setGeometry(QRect(737, 860, 93, 28))
        self.cancelButton.setFont(font)
        cancel_style = button_style.format(
            border_color="rgb(224, 9, 22)",
            background_color="rgb(224, 9, 22)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(172, 16, 12)",
            pressed_color="rgb(224, 54, 80)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        self.cancelButton.setStyleSheet(cancel_style)
        self.cancelButton.clicked.connect(lambda: self.close())

        # Отображение окна
        self.show()

        self.raise_()

    'SUBMIT HANDLER'
    def onSubmit(self):
        'Get the drawing of path'
        path = self.eventhandler.getPath()
        self.image = QImage(800, 800, QImage.Format_RGB32)
        self.image.fill(Qt.black)
        painter = QPainter(self.image)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        painter.setPen(Qt.white)
        painter.fillPath(path, QBrush(QColor("white")))
        painter.drawPath(path)

        self.submited.emit(self.slice_edit)
        'Save path'
        filePath = f'result_folder/{patient}/contours/{self.patient_edit}_{int(self.slice_edit) - 1}.png'
        self.image.save(filePath)
        'Rescale path to original size of dicom images'
        self.rescaleImage()
        'Save edited image'
        self.combineImages()
        'Clear path'
        self.eventhandler.clearPath()
        'Close the second window'
        self.close()

    def rescaleImage(self):
        filePath = f'result_folder/{patient}/contours/{self.patient_edit}_{int(self.slice_edit) - 1}.png'
        img = cv.imread(filePath)
        dim = (512, 512)
        resized = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        cv.imwrite(filePath, resized)

    def combineImages(self):
        mask_rgb = cv.imread(f'result_folder/{patient}/contours/{self.patient_edit}_{int(self.slice_edit) - 1}.png')
        img = cv.imread(f'result_folder/{patient}/slices/{self.patient_edit}_{int(self.slice_edit) - 1}.png')
        mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [20, 14, 196]
        out = cv.addWeighted(mask_rgb, 0.2, img, 1, 0, img)
        filepath = f'result_folder/{patient}/combined/{self.patient_edit}_{int(self.slice_edit) - 1}.png'
        cv.imwrite(filepath, out)

class EventHandler(QLabel):
    flag = True
    points = []
    path = QtGui.QPainterPath()

    def mousePressEvent(self, event):
        self.points.append(event.pos())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag:
            self.points = []
            self.path = QtGui.QPainterPath()
            self.flag = False
        painter = QPainter(self)
        'Pen of the points'
        point_color = QColor(20, 14, 196)  # Бордовый цвет
        point_size = 15
        painter.setPen(QPen(point_color, point_size, Qt.SolidLine, Qt.RoundCap))
        'Draw all points'
        for pos in self.points:
            painter.drawPoint(pos)
        'Build the path'
        curve_color = QColor(0, 206, 209)  # Берюзовый цвет
        curve_size = 3
        if len(self.points) > 2:
            self.buildPath()
            'Pen of the line'
            painter.setPen(QPen(curve_color, curve_size, Qt.SolidLine))
            painter.drawPath(self.path)

    'PATH HANDLER'
    def buildPath(self):
        factor = 0.5
        self.path = QtGui.QPainterPath(self.points[0])
        for p, current in enumerate(self.points[1:-1], 1):
            'Previous segment'
            source = QtCore.QLineF(self.points[p - 1], current)
            'Next segment'
            target = QtCore.QLineF(current, self.points[p + 1])
            targetAngle = target.angleTo(source)
            if targetAngle > 180:
                angle = (source.angle() + source.angleTo(target) / 2) % 360
            else:
                angle = (target.angle() + target.angleTo(source) / 2) % 360

            revTarget = QtCore.QLineF.fromPolar(source.length() * factor, angle + 180).translated(current)
            cp2 = revTarget.p2()

            if p == 1:
                self.path.quadTo(cp2, current)
            else:
                'Use the control point "cp1" set in the * previous * cycle'
                self.path.cubicTo(cp1, cp2, current)

            revSource = QtCore.QLineF.fromPolar(target.length() * factor, angle).translated(current)
            cp1 = revSource.p2()

        'The final curve, that joins to the last point'
        self.path.quadTo(cp1, self.points[-1])


# def buildPath(self):
#     factor = 0.5
#     self.path = QtGui.QPainterPath(self.points[0])

#     for p, current in enumerate(self.points[1:-1], 1):
#         'Previous segment'
#         source = QtCore.QLineF(self.points[p - 1], current)
#         'Next segment'
#         target = QtCore.QLineF(current, self.points[p + 1])
        
#         angle = (target.angle() + target.angleTo(source) / 2) % 360

#         revTarget = QtCore.QLineF.fromPolar(source.length() * factor, angle + 180).translated(current)
#         cp2 = revTarget.p2()

#         if p == 1:
#             # Интерполяция первой точки с использованием кубического сплайна
#             cp1 = current + (cp2 - current) * factor
#             self.path.cubicTo(cp1, cp2, current)
#         else:
#             'Use the control point "cp1" set in the * previous * cycle'
#             self.path.cubicTo(cp1, cp2, current)

#         revSource = QtCore.QLineF.fromPolar(target.length() * factor, angle).translated(current)
#         cp1 = revSource.p2()

#     'The final curve, that joins to the last point'
#     # Интерполяция последней точки с использованием кубического сплайна
#     self.path.cubicTo(cp1, self.points[-1], self.points[-1])


    def getPath(self):
        return self.path

    def clearPath(self):
        self.flag = True


'***************************************************************************************************************************************************'
'THIRD WINDOW'
'***************************************************************************************************************************************************'


class ThirdWindow(QMainWindow):
    def __init__(self, volume_type):
        super().__init__()
        self.img_label = QLabel(self)
        self.volume_type = volume_type
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("3D Epicardial Fat")
        self.setStyleSheet("background-color:rgb(255, 255, 255);")
        self.setWindowIcon(QtGui.QIcon(r'.\resources\logo_w.png'))

        # Set up the 3D plot based on the volume type
        if self.volume_type == "3d_fat":
            savePath = self.generate3DFatVolume(patient, self.volume_type)
        elif self.volume_type == "new_3d_fat":
            savePath = self.generate3DFatVolume(patient, self.volume_type)

        img = cv.imread(savePath)  # Load the saved 3D plot image
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.img_label.setPixmap(pixmap)

        # Create a central widget and set it for the QMainWindow
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        # Add the QLabel with the image to the layout
        self.img_label = QLabel(central_widget)
        self.img_label.setPixmap(pixmap)
        layout.addWidget(self.img_label)

        'Size of window'
        self.resize(860, 920)

        'Image properties'
        self.img_label.setMinimumSize(QtCore.QSize(800, 800))
        self.img_label.setBaseSize(QtCore.QSize(512, 512))
        self.img_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.img_label.setScaledContents(True)
        self.img_label.setGeometry(QRect(30, 30, width, height))
        self.img_label.setCursor(Qt.ArrowCursor)

        # Создание стилей для кнопок
        font = QtGui.QFont()
        font.setPointSize(9)
        button_style = """
            QPushButton {{
                border: 2px solid {border_color};
                border-radius: 10px;
                background-color: {background_color};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-color: {border_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {pressed_border_color};
            }}
        """

        # Установка свойств кнопки "Cancel"
        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.setGeometry(QRect(737, 860, 93, 28))
        self.cancelButton.setFont(font)
        cancel_style = button_style.format(
            border_color="rgb(224, 9, 22)",
            background_color="rgb(224, 9, 22)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(172, 16, 12)",
            pressed_color="rgb(224, 54, 80)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        self.cancelButton.setStyleSheet(cancel_style)
        self.cancelButton.clicked.connect(lambda: self.close())

        # Отображение окна
        self.show()

        self.raise_()

    def generate3DFatVolume(self, patient, volume_type):
        filePath = os.path.join('result_folder', patient, 'fat')
        savePath = os.path.join('result_folder', patient, f'{volume_type}.png')
        plt_plot_3d(filePath, save_path=savePath)

        return savePath


'***************************************************************************************************************************************************'
'FOURTH WINDOW'
'***************************************************************************************************************************************************'


class FourthWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.results = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Your Window Title")
        self.setStyleSheet("background-color:rgb(255, 255, 255);")

        # Создание таблицы для отображения результатов
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setGeometry(30, 30, 600, 600)

        # Создание стилей для кнопок
        font = QtGui.QFont()
        font.setPointSize(9)
        button_style = """
            QPushButton {{
                border: 2px solid {border_color};
                border-radius: 10px;
                background-color: {background_color};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-color: {border_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {pressed_border_color};
            }}
        """

        # Замена Done на Save
        self.saveButton = QPushButton('Save', self)
        self.saveButton.setGeometry(QRect(624, 860, 93, 28))
        self.saveButton.setFont(font)
        save_style = button_style.format(
            border_color="rgb(0, 128, 0)",
            background_color="rgb(0, 128, 0)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(0, 100, 0)",
            pressed_color="rgb(0, 200, 0)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        self.saveButton.setStyleSheet(save_style)
        self.saveButton.clicked.connect(self.onSave)

        # Установка свойств кнопки "Cancel"
        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.setGeometry(QRect(737, 860, 93, 28))
        self.cancelButton.setFont(font)
        cancel_style = button_style.format(
            border_color="rgb(224, 9, 22)",
            background_color="rgb(224, 9, 22)",
            text_color="rgb(234, 234, 234)",
            hover_color="rgb(172, 16, 12)",
            pressed_color="rgb(224, 54, 80)",
            pressed_border_color="rgb(100, 100, 100)"
        )
        self.cancelButton.setStyleSheet(cancel_style)
        self.cancelButton.clicked.connect(lambda: self.close())

        # Заполнение таблицы данными
        self.populateTableWidget(self.results)

        # Отображение окна по центру
        self.centerWindow()

        # Отображение окна
        self.show()

    def centerWindow(self):
        frameGm = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def populateTableWidget(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.tableWidget.setItem(i, j, item)

    def onSave(self):
        save_csv_path = os.path.join('result_folder', patient, 'radiomics_features.csv')

        if save_csv_path:
            self.results.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())