# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Project_Design_v2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1258, 712)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 1241, 661))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.spinBox_2 = QtWidgets.QSpinBox(self.tab)
        self.spinBox_2.setGeometry(QtCore.QRect(440, 220, 62, 32))
        self.spinBox_2.setMinimum(3)
        self.spinBox_2.setMaximum(48)
        self.spinBox_2.setProperty("value", 12)
        self.spinBox_2.setObjectName("spinBox_2")
        self.line_5 = QtWidgets.QFrame(self.tab)
        self.line_5.setGeometry(QtCore.QRect(640, 500, 451, 21))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(400, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 340, 121, 29))
        self.radioButton_2.setIconSize(QtCore.QSize(24, 24))
        self.radioButton_2.setObjectName("radioButton_2")
        self.startDate = QtWidgets.QDateEdit(self.tab)
        self.startDate.setGeometry(QtCore.QRect(230, 40, 110, 32))
        self.startDate.setDate(QtCore.QDate(2010, 1, 1))
        self.startDate.setCalendarPopup(True)
        self.startDate.setObjectName("startDate")
        self.line_2 = QtWidgets.QFrame(self.tab)
        self.line_2.setGeometry(QtCore.QRect(180, 10, 20, 91))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(30, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.Graph_Update = QtWidgets.QPushButton(self.tab)
        self.Graph_Update.setGeometry(QtCore.QRect(1030, 540, 131, 51))
        self.Graph_Update.setObjectName("Graph_Update")
        self.line_4 = QtWidgets.QFrame(self.tab)
        self.line_4.setGeometry(QtCore.QRect(530, 10, 20, 541))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_16 = QtWidgets.QLabel(self.tab)
        self.label_16.setGeometry(QtCore.QRect(270, 520, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_13 = QtWidgets.QLabel(self.tab)
        self.label_13.setGeometry(QtCore.QRect(790, 530, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.line = QtWidgets.QFrame(self.tab)
        self.line.setGeometry(QtCore.QRect(10, 160, 521, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.mpl = MplWidget(self.tab)
        self.mpl.setGeometry(QtCore.QRect(550, 10, 671, 491))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(sizePolicy)
        self.mpl.setObjectName("mpl")
        self.label_12 = QtWidgets.QLabel(self.tab)
        self.label_12.setGeometry(QtCore.QRect(20, 460, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.line_3 = QtWidgets.QFrame(self.tab)
        self.line_3.setGeometry(QtCore.QRect(10, 500, 521, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.maxW = QtWidgets.QDoubleSpinBox(self.tab)
        self.maxW.setGeometry(QtCore.QRect(110, 220, 71, 32))
        self.maxW.setDecimals(4)
        self.maxW.setMaximum(1.0)
        self.maxW.setSingleStep(0.0001)
        self.maxW.setProperty("value", 0.1)
        self.maxW.setObjectName("maxW")
        self.Button_Add_Port = QtWidgets.QPushButton(self.tab)
        self.Button_Add_Port.setGeometry(QtCore.QRect(430, 320, 81, 51))
        self.Button_Add_Port.setObjectName("Button_Add_Port")
        self.label_15 = QtWidgets.QLabel(self.tab)
        self.label_15.setGeometry(QtCore.QRect(290, 330, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.line_6 = QtWidgets.QFrame(self.tab)
        self.line_6.setGeometry(QtCore.QRect(120, 310, 21, 51))
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.Graph_startDate = QtWidgets.QDateEdit(self.tab)
        self.Graph_startDate.setGeometry(QtCore.QRect(620, 560, 110, 32))
        self.Graph_startDate.setDate(QtCore.QDate(2010, 1, 1))
        self.Graph_startDate.setCalendarPopup(True)
        self.Graph_startDate.setObjectName("Graph_startDate")
        self.label_6 = QtWidgets.QLabel(self.tab)
        self.label_6.setGeometry(QtCore.QRect(120, 260, 52, 16))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setItalic(True)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.spinBox = QtWidgets.QSpinBox(self.tab)
        self.spinBox.setGeometry(QtCore.QRect(280, 220, 62, 32))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(6)
        self.spinBox.setObjectName("spinBox")
        self.label_7 = QtWidgets.QLabel(self.tab)
        self.label_7.setGeometry(QtCore.QRect(210, 180, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.radioButton_4 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_4.setGeometry(QtCore.QRect(150, 340, 111, 29))
        self.radioButton_4.setIconSize(QtCore.QSize(24, 24))
        self.radioButton_4.setObjectName("radioButton_4")
        self.label_17 = QtWidgets.QLabel(self.tab)
        self.label_17.setGeometry(QtCore.QRect(20, 280, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.radioButton_3 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_3.setGeometry(QtCore.QRect(150, 310, 111, 29))
        self.radioButton_3.setIconSize(QtCore.QSize(24, 24))
        self.radioButton_3.setObjectName("radioButton_3")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(20, 180, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.Button_Import_Index = QtWidgets.QPushButton(self.tab)
        self.Button_Import_Index.setGeometry(QtCore.QRect(130, 80, 51, 41))
        self.Button_Import_Index.setObjectName("Button_Import_Index")
        self.UniverseSelect = QtWidgets.QComboBox(self.tab)
        self.UniverseSelect.setGeometry(QtCore.QRect(30, 40, 121, 31))
        self.UniverseSelect.setObjectName("UniverseSelect")
        self.UniverseSelect.addItem("")
        self.UniverseSelect.addItem("")
        self.UniverseSelect.addItem("")
        self.Button_Import_Data = QtWidgets.QPushButton(self.tab)
        self.Button_Import_Data.setGeometry(QtCore.QRect(10, 80, 51, 41))
        self.Button_Import_Data.setObjectName("Button_Import_Data")
        self.Graph_endDate = QtWidgets.QDateEdit(self.tab)
        self.Graph_endDate.setGeometry(QtCore.QRect(790, 560, 110, 32))
        self.Graph_endDate.setDate(QtCore.QDate(2015, 1, 1))
        self.Graph_endDate.setCalendarPopup(True)
        self.Graph_endDate.setObjectName("Graph_endDate")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(50, 520, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(230, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Button_Import_Compo = QtWidgets.QPushButton(self.tab)
        self.Button_Import_Compo.setGeometry(QtCore.QRect(70, 80, 51, 41))
        self.Button_Import_Compo.setObjectName("Button_Import_Compo")
        self.label_8 = QtWidgets.QLabel(self.tab)
        self.label_8.setGeometry(QtCore.QRect(20, 120, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.radioButton = QtWidgets.QRadioButton(self.tab)
        self.radioButton.setGeometry(QtCore.QRect(10, 310, 111, 29))
        self.radioButton.setIconSize(QtCore.QSize(24, 24))
        self.radioButton.setObjectName("radioButton")
        self.label_14 = QtWidgets.QLabel(self.tab)
        self.label_14.setGeometry(QtCore.QRect(620, 530, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_11 = QtWidgets.QLabel(self.tab)
        self.label_11.setGeometry(QtCore.QRect(450, 260, 52, 16))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setItalic(True)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setGeometry(QtCore.QRect(290, 260, 52, 16))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setItalic(True)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.minW = QtWidgets.QDoubleSpinBox(self.tab)
        self.minW.setGeometry(QtCore.QRect(20, 220, 71, 32))
        self.minW.setDecimals(4)
        self.minW.setSingleStep(0.0001)
        self.minW.setObjectName("minW")
        self.progressBar = QtWidgets.QProgressBar(self.tab)
        self.progressBar.setGeometry(QtCore.QRect(10, 590, 231, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.Button_Reset = QtWidgets.QPushButton(self.tab)
        self.Button_Reset.setGeometry(QtCore.QRect(420, 520, 81, 31))
        self.Button_Reset.setObjectName("Button_Reset")
        self.endDate = QtWidgets.QDateEdit(self.tab)
        self.endDate.setGeometry(QtCore.QRect(400, 40, 110, 32))
        self.endDate.setDate(QtCore.QDate(2015, 1, 1))
        self.endDate.setCalendarPopup(True)
        self.endDate.setObjectName("endDate")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(30, 260, 52, 16))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setItalic(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_10 = QtWidgets.QLabel(self.tab)
        self.label_10.setGeometry(QtCore.QRect(420, 180, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_33 = QtWidgets.QLabel(self.tab)
        self.label_33.setGeometry(QtCore.QRect(80, 380, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setItalic(True)
        self.label_33.setFont(font)
        self.label_33.setObjectName("label_33")
        self.Button_Save = QtWidgets.QPushButton(self.tab)
        self.Button_Save.setGeometry(QtCore.QRect(420, 570, 81, 31))
        self.Button_Save.setObjectName("Button_Save")
        self.label_38 = QtWidgets.QLabel(self.tab)
        self.label_38.setGeometry(QtCore.QRect(290, 560, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_38.setFont(font)
        self.label_38.setScaledContents(False)
        self.label_38.setObjectName("label_38")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(180, 50, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.Portfolio_Select = QtWidgets.QComboBox(self.tab_2)
        self.Portfolio_Select.setGeometry(QtCore.QRect(10, 40, 121, 31))
        self.Portfolio_Select.setObjectName("Portfolio_Select")
        self.label_19 = QtWidgets.QLabel(self.tab_2)
        self.label_19.setGeometry(QtCore.QRect(10, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.tab_2)
        self.label_20.setGeometry(QtCore.QRect(180, 0, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.endDate_2 = QtWidgets.QDateEdit(self.tab_2)
        self.endDate_2.setGeometry(QtCore.QRect(170, 70, 110, 32))
        self.endDate_2.setDate(QtCore.QDate(2015, 1, 1))
        self.endDate_2.setCalendarPopup(True)
        self.endDate_2.setObjectName("endDate_2")
        self.startDate_2 = QtWidgets.QDateEdit(self.tab_2)
        self.startDate_2.setGeometry(QtCore.QRect(170, 20, 110, 32))
        self.startDate_2.setDate(QtCore.QDate(2010, 1, 1))
        self.startDate_2.setCalendarPopup(True)
        self.startDate_2.setObjectName("startDate_2")
        self.label_21 = QtWidgets.QLabel(self.tab_2)
        self.label_21.setGeometry(QtCore.QRect(10, 130, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.line_7 = QtWidgets.QFrame(self.tab_2)
        self.line_7.setGeometry(QtCore.QRect(10, 150, 391, 21))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_22 = QtWidgets.QLabel(self.tab_2)
        self.label_22.setGeometry(QtCore.QRect(10, 220, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.tab_2)
        self.label_23.setGeometry(QtCore.QRect(10, 260, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.tab_2)
        self.label_24.setGeometry(QtCore.QRect(10, 300, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.tab_2)
        self.label_25.setGeometry(QtCore.QRect(10, 340, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.tab_2)
        self.label_26.setGeometry(QtCore.QRect(10, 420, 52, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.tab_2)
        self.label_27.setGeometry(QtCore.QRect(160, 180, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.tab_2)
        self.label_28.setGeometry(QtCore.QRect(290, 180, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.line_8 = QtWidgets.QFrame(self.tab_2)
        self.line_8.setGeometry(QtCore.QRect(10, 200, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_8.setFont(font)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.line_9 = QtWidgets.QFrame(self.tab_2)
        self.line_9.setGeometry(QtCore.QRect(10, 240, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_9.setFont(font)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.line_10 = QtWidgets.QFrame(self.tab_2)
        self.line_10.setGeometry(QtCore.QRect(10, 280, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_10.setFont(font)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.line_11 = QtWidgets.QFrame(self.tab_2)
        self.line_11.setGeometry(QtCore.QRect(10, 320, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_11.setFont(font)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.line_12 = QtWidgets.QFrame(self.tab_2)
        self.line_12.setGeometry(QtCore.QRect(10, 360, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_12.setFont(font)
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.line_13 = QtWidgets.QFrame(self.tab_2)
        self.line_13.setGeometry(QtCore.QRect(10, 440, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_13.setFont(font)
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.label_29 = QtWidgets.QLabel(self.tab_2)
        self.label_29.setGeometry(QtCore.QRect(10, 380, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.line_14 = QtWidgets.QFrame(self.tab_2)
        self.line_14.setGeometry(QtCore.QRect(10, 400, 351, 16))
        font = QtGui.QFont()
        font.setPointSize(2)
        font.setBold(False)
        font.setWeight(50)
        self.line_14.setFont(font)
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_15 = QtWidgets.QFrame(self.tab_2)
        self.line_15.setGeometry(QtCore.QRect(420, 10, 16, 591))
        self.line_15.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.mpl_returns = MplSubPlotWidget(self.tab_2)
        self.mpl_returns.setGeometry(QtCore.QRect(440, 40, 411, 581))
        self.mpl_returns.setObjectName("mpl_returns")
        self.mpl_vol = MplWidget(self.tab_2)
        self.mpl_vol.setGeometry(QtCore.QRect(860, 40, 371, 261))
        self.mpl_vol.setObjectName("mpl_vol")
        self.mpl_distrib = MplWidget(self.tab_2)
        self.mpl_distrib.setGeometry(QtCore.QRect(860, 400, 371, 221))
        self.mpl_distrib.setObjectName("mpl_distrib")
        self.label_30 = QtWidgets.QLabel(self.tab_2)
        self.label_30.setGeometry(QtCore.QRect(440, 10, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.tab_2)
        self.label_31.setGeometry(QtCore.QRect(860, 10, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.tab_2)
        self.label_32.setGeometry(QtCore.QRect(860, 370, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.spinBox_volatility = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_volatility.setGeometry(QtCore.QRect(1020, 310, 71, 32))
        self.spinBox_volatility.setObjectName("spinBox_volatility")
        self.label_34 = QtWidgets.QLabel(self.tab_2)
        self.label_34.setGeometry(QtCore.QRect(930, 320, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.tab_2)
        self.label_35.setGeometry(QtCore.QRect(1110, 320, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.port_returns_label = QtWidgets.QLabel(self.tab_2)
        self.port_returns_label.setGeometry(QtCore.QRect(160, 220, 51, 16))
        self.port_returns_label.setObjectName("port_returns_label")
        self.port_volatility_label = QtWidgets.QLabel(self.tab_2)
        self.port_volatility_label.setGeometry(QtCore.QRect(160, 260, 51, 16))
        self.port_volatility_label.setObjectName("port_volatility_label")
        self.port_Sharpe_label = QtWidgets.QLabel(self.tab_2)
        self.port_Sharpe_label.setGeometry(QtCore.QRect(160, 300, 51, 16))
        self.port_Sharpe_label.setObjectName("port_Sharpe_label")
        self.bck_returns_label = QtWidgets.QLabel(self.tab_2)
        self.bck_returns_label.setGeometry(QtCore.QRect(290, 220, 51, 16))
        self.bck_returns_label.setObjectName("bck_returns_label")
        self.bck_volatility_label = QtWidgets.QLabel(self.tab_2)
        self.bck_volatility_label.setGeometry(QtCore.QRect(290, 260, 51, 16))
        self.bck_volatility_label.setObjectName("bck_volatility_label")
        self.bck_Sharpe_label = QtWidgets.QLabel(self.tab_2)
        self.bck_Sharpe_label.setGeometry(QtCore.QRect(290, 300, 51, 16))
        self.bck_Sharpe_label.setObjectName("bck_Sharpe_label")
        self.port_Beta_label = QtWidgets.QLabel(self.tab_2)
        self.port_Beta_label.setGeometry(QtCore.QRect(160, 420, 51, 16))
        self.port_Beta_label.setObjectName("port_Beta_label")
        self.bck_Beta_label = QtWidgets.QLabel(self.tab_2)
        self.bck_Beta_label.setGeometry(QtCore.QRect(290, 420, 51, 16))
        self.bck_Beta_label.setObjectName("bck_Beta_label")
        self.spinBox_drawdown = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_drawdown.setGeometry(QtCore.QRect(1020, 340, 71, 32))
        self.spinBox_drawdown.setMinimum(1)
        self.spinBox_drawdown.setMaximum(2000)
        self.spinBox_drawdown.setSingleStep(10)
        self.spinBox_drawdown.setProperty("value", 90)
        self.spinBox_drawdown.setObjectName("spinBox_drawdown")
        self.label_36 = QtWidgets.QLabel(self.tab_2)
        self.label_36.setGeometry(QtCore.QRect(930, 350, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_36.setFont(font)
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.tab_2)
        self.label_37.setGeometry(QtCore.QRect(1110, 350, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.port_DD_label = QtWidgets.QLabel(self.tab_2)
        self.port_DD_label.setGeometry(QtCore.QRect(160, 340, 51, 16))
        self.port_DD_label.setObjectName("port_DD_label")
        self.bck_DD_label = QtWidgets.QLabel(self.tab_2)
        self.bck_DD_label.setGeometry(QtCore.QRect(290, 340, 51, 16))
        self.bck_DD_label.setObjectName("bck_DD_label")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1258, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "End Date"))
        self.radioButton_2.setText(_translate("MainWindow", "Equal Weighted"))
        self.startDate.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.label.setText(_translate("MainWindow", "Universe"))
        self.Graph_Update.setText(_translate("MainWindow", "Refresh"))
        self.label_16.setText(_translate("MainWindow", "Reset the Backtest List"))
        self.label_13.setText(_translate("MainWindow", "End Date"))
        self.label_12.setText(_translate("MainWindow", "Backtest Parameters"))
        self.Button_Add_Port.setText(_translate("MainWindow", "OK"))
        self.label_15.setText(_translate("MainWindow", "Add to Backtest List"))
        self.Graph_startDate.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.label_6.setText(_translate("MainWindow", "Maximum"))
        self.label_7.setText(_translate("MainWindow", "Rebalancing Frequency"))
        self.radioButton_4.setText(_translate("MainWindow", "1 / Sigma"))
        self.label_17.setText(_translate("MainWindow", "Strategy "))
        self.radioButton_3.setText(_translate("MainWindow", "1 / Beta"))
        self.label_4.setText(_translate("MainWindow", "Weight Constraints"))
        self.Button_Import_Index.setText(_translate("MainWindow", "Index"))
        self.UniverseSelect.setItemText(0, _translate("MainWindow", "MSCI EU"))
        self.UniverseSelect.setItemText(1, _translate("MainWindow", "MSCI Europe"))
        self.UniverseSelect.setItemText(2, _translate("MainWindow", "STOXX 50"))
        self.Button_Import_Data.setText(_translate("MainWindow", "Data"))
        self.Graph_endDate.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.pushButton.setText(_translate("MainWindow", "Launch"))
        self.label_2.setText(_translate("MainWindow", "Start Date"))
        self.Button_Import_Compo.setText(_translate("MainWindow", "Compo"))
        self.label_8.setText(_translate("MainWindow", "Portfolio Parameters"))
        self.radioButton.setText(_translate("MainWindow", "Min Variance"))
        self.label_14.setText(_translate("MainWindow", "Start Date"))
        self.label_11.setText(_translate("MainWindow", "Months"))
        self.label_9.setText(_translate("MainWindow", "Months"))
        self.Button_Reset.setText(_translate("MainWindow", "Reset"))
        self.endDate.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.label_5.setText(_translate("MainWindow", "Minimum"))
        self.label_10.setText(_translate("MainWindow", "Historic Length"))
        self.label_33.setText(_translate("MainWindow", "... More to Come"))
        self.Button_Save.setText(_translate("MainWindow", "Save"))
        self.label_38.setText(_translate("MainWindow", "Save the Backtest"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Settings"))
        self.label_18.setText(_translate("MainWindow", "To"))
        self.label_19.setText(_translate("MainWindow", "Portoflio"))
        self.label_20.setText(_translate("MainWindow", "From"))
        self.endDate_2.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.startDate_2.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.label_21.setText(_translate("MainWindow", "Risk Metrics"))
        self.label_22.setText(_translate("MainWindow", "Returns"))
        self.label_23.setText(_translate("MainWindow", "Volatility"))
        self.label_24.setText(_translate("MainWindow", "Sharpe Ratio"))
        self.label_25.setText(_translate("MainWindow", "Max Drawdown"))
        self.label_26.setText(_translate("MainWindow", "Beta"))
        self.label_27.setText(_translate("MainWindow", "Portfolio"))
        self.label_28.setText(_translate("MainWindow", "Benchmark"))
        self.label_29.setText(_translate("MainWindow", "Recovery"))
        self.label_30.setText(_translate("MainWindow", "Returns Analysis"))
        self.label_31.setText(_translate("MainWindow", "Vol Analysis"))
        self.label_32.setText(_translate("MainWindow", "Distribution"))
        self.label_34.setText(_translate("MainWindow", "Volatility"))
        self.label_35.setText(_translate("MainWindow", "Days"))
        self.port_returns_label.setText(_translate("MainWindow", "0"))
        self.port_volatility_label.setText(_translate("MainWindow", "-"))
        self.port_Sharpe_label.setText(_translate("MainWindow", "0"))
        self.bck_returns_label.setText(_translate("MainWindow", "0"))
        self.bck_volatility_label.setText(_translate("MainWindow", "-"))
        self.bck_Sharpe_label.setText(_translate("MainWindow", "0"))
        self.port_Beta_label.setText(_translate("MainWindow", "0"))
        self.bck_Beta_label.setText(_translate("MainWindow", "0"))
        self.label_36.setText(_translate("MainWindow", "Drawdown"))
        self.label_37.setText(_translate("MainWindow", "Days"))
        self.port_DD_label.setText(_translate("MainWindow", "0"))
        self.bck_DD_label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Risk Analysis"))

from mplsubplotwidget import MplSubPlotWidget
from mplwidget import MplWidget