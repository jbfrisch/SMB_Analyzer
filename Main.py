#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:49:12 2017

@author: wasp
"""
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon

import sys
import pickle
import datetime
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from pandas.tseries.offsets import *
#from threading import *

import PortfolioOptimization.GUI.design6 as design
import PortfolioOptimization.GUI.Benchmark as bch
import PortfolioOptimization.GUI.Portfolio as port

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import PortfolioOptimization.utils.po_theme as th
import PortfolioOptimization.utils.Drawdown as DD
#import PortfolioOptimization.utils.Opt_Histogram as oph

try:
    import PortfolioOptimization.utils.blpapiwrapper as myblp
except ImportError as error:
    print(error.message)
except Exception as exception:
    print(exception, False)

class MainWindow(QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        
        self.Portfolio_Benchmark = bch.Benchmark()
        self.Portfolio_List = []
        self.Portfolio_Backtest = port.Portfolio()
        self.Bench_Strategy = pd.DataFrame()
        #
        if sys.platform == 'win32':
            self.ref_eqty = pickle.load(open('.//AppData//ref_eqty_Win.pickle', 'rb'))
        else:
            self.ref_eqty = pickle.load(open('.//AppData//ref_eqty.pickle', 'rb'))
        #
        self.actionImport_Portfolio.triggered.connect(self.Import_Portfolio)
        self.actionRefresh_Context.triggered.connect(self.Refresh_Backtest_Context)
        
        self.pushButton.clicked.connect(self.launcher_On_CLick)
        self.Button_Add_Port.clicked.connect(self.Add_To_Backtest_List)
        self.Button_Reset.clicked.connect(self.Reset_Backtest_List)
        self.Button_Save.clicked.connect(self.Save_as_Pickle)
        self.Button_Import_Data.clicked.connect(self.import_Historic_Data)
        self.Button_Import_Compo.clicked.connect(self.import_Historic_Compo)
        self.Button_Import_Index.clicked.connect(self.import_Historic_Index)
        self.UniverseSelect.currentIndexChanged.connect(self.Universe_Selection)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_Date_Picker)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_Returns)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_Volatility)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_TrackingError)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_Sharpe_Ratio)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_Beta)
        self.Portfolio_Select.currentIndexChanged.connect(self.Update_MaxDD)
        self.Portfolio_Select.currentIndexChanged.connect(self.Plot_Perf_Analysis)
        self.Portfolio_Select.currentIndexChanged.connect(self.Plot_Distribution)
        self.Portfolio_Select.currentIndexChanged.connect(self.Plot_Volatility)
        self.Portfolio_Select_2.currentIndexChanged.connect(self.Plot_Sectorial_Repartition)
        self.Portfolio_Select_2.currentIndexChanged.connect(self.Plot_Country_Repartition)
        self.Portfolio_Select_2.currentIndexChanged.connect(self.Plot_Turnover)
        self.startDate_2.dateChanged.connect(self.Update_Returns)
        self.startDate_2.dateChanged.connect(self.Update_Volatility)
        self.startDate_2.dateChanged.connect(self.Update_TrackingError)
        self.startDate_2.dateChanged.connect(self.Update_Sharpe_Ratio)
        self.startDate_2.dateChanged.connect(self.Update_Beta)
        self.startDate_2.dateChanged.connect(self.Update_MaxDD)
        self.startDate_2.dateChanged.connect(self.Plot_Perf_Analysis)
        self.startDate_2.dateChanged.connect(self.Plot_Distribution)
        self.startDate_2.dateChanged.connect(self.Plot_Volatility)
        self.startDate_2.dateChanged.connect(self.Plot_Sectorial_Repartition)
        self.startDate_2.dateChanged.connect(self.Plot_Country_Repartition)
        self.startDate_2.dateChanged.connect(self.Plot_Turnover)
        self.startDate_3.dateChanged.connect(self.Plot_Sectorial_Repartition)
        self.startDate_3.dateChanged.connect(self.Plot_Country_Repartition)
        self.startDate_3.dateChanged.connect(self.Plot_Turnover)
        self.endDate_2.dateChanged.connect(self.Update_Returns)
        self.endDate_2.dateChanged.connect(self.Update_Volatility)
        self.endDate_2.dateChanged.connect(self.Update_TrackingError)
        self.endDate_2.dateChanged.connect(self.Update_Sharpe_Ratio)
        self.endDate_2.dateChanged.connect(self.Update_Beta)
        self.endDate_2.dateChanged.connect(self.Update_MaxDD)
        self.endDate_2.dateChanged.connect(self.Plot_Perf_Analysis)
        self.endDate_2.dateChanged.connect(self.Plot_Distribution)
        self.endDate_2.dateChanged.connect(self.Plot_Volatility)
        self.endDate_3.dateChanged.connect(self.Plot_Sectorial_Repartition)
        self.endDate_3.dateChanged.connect(self.Plot_Country_Repartition)
        self.endDate_3.dateChanged.connect(self.Plot_Turnover)
        self.spinBox_drawdown.valueChanged.connect(self.Plot_Perf_Analysis)
        self.spinBox_volatility.valueChanged.connect(self.Plot_Volatility)
        self.Graph_Update.clicked.connect(self.Refresh_plot)
    
    def Update_PortfolioSelect_List(self):
        i=1
        nameList = []
        self.Portfolio_Select.clear()
        self.Portfolio_Select_2.clear()
        for portBck in self.Portfolio_List:
            nameList.append("%s_%d"%(portBck.Opt_Strat,i))
            i+=1
        self.Portfolio_Select.addItems(nameList)
        self.Portfolio_Select_2.addItems(nameList)
    
    def Update_Date_Picker(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            
            self.startDate_2.minimumDate = QDate(self.Portfolio_List[port_Idx].Dates[0].year,
                                                   self.Portfolio_List[port_Idx].Dates[0].month,
                                                   self.Portfolio_List[port_Idx].Dates[0].day)
            
            self.endDate_2.maximumDate = QDate(self.Portfolio_List[port_Idx].Dates[1].year,
                                               self.Portfolio_List[port_Idx].Dates[1].month,
                                               self.Portfolio_List[port_Idx].Dates[1].day)
            
            self.startDate_2.setDate(self.startDate_2.minimumDate)
            self.endDate_2.setDate(self.endDate_2.maximumDate)
    
    def Update_Returns(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() 
            idx_Min = self.startDate_2.dateTime().toPyDateTime() 
            
            p_returns = (np.prod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], axis=0) - 1.0) * 100.0
            b_returns = (np.prod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], axis=0) - 1.0) * 100.0
            
            self.port_returns_label.setNum(p_returns)
            self.bck_returns_label.setNum(b_returns)

    def Update_Volatility(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() 
            idx_Min = self.startDate_2.dateTime().toPyDateTime() 
            
            p_vol = np.std(self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], ddof=1) 
            b_vol = np.std(self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], ddof=1)
            
            self.port_volatility_label.setNum(p_vol * np.sqrt(250.0) * 100.0)
            self.bck_volatility_label.setNum(b_vol * np.sqrt(250.0) * 100.0)        
    
    def Update_TrackingError(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() 
            idx_Min = self.startDate_2.dateTime().toPyDateTime()

            p_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]]
            b_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index]
            
            self.port_TE_label.setNum(np.std(p_returns - b_returns, ddof=1) * np.sqrt(250.0) * 100.0)
            
    
    def Update_Sharpe_Ratio(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() 
            idx_Min = self.startDate_2.dateTime().toPyDateTime() 
            
            p_returns = (np.prod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], axis=0) - 1.0) * 100.0
            p_vol = np.std(self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], ddof=1) * np.sqrt(250.0) * 100.0
            
            b_returns = (np.prod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], axis=0) - 1.0) * 100.0
            b_vol = np.std(self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], ddof=1) * np.sqrt(250.0) * 100.0
            
            self.port_Sharpe_label.setNum(p_returns / p_vol)
            self.bck_Sharpe_label.setNum(b_returns / b_vol)

    def Update_Beta(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() 
            idx_Min = self.startDate_2.dateTime().toPyDateTime() 
            
            p_cov = np.cov(self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], 
                           self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index],
                           ddof=1)[0][1] 
            b_vol = np.var(self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], ddof=1)
            
            self.port_Beta_label.setNum(p_cov / b_vol)
            self.bck_Beta_label.setNum(1.0)
    
    def Update_MaxDD(self):
        if self.Portfolio_Select.count() > 0:
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.max()
            idx_Min = self.startDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.min()
            window_length = self.spinBox_drawdown.value()
            
            perf_Port = (np.cumprod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], axis=0) - 1.0) * 100.0
            perf_Bench = (np.cumprod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], axis=0) - 1.0) * 100.0
            
#            rolling_dd_port = perf_Port.rolling(window_length, min_periods=0).apply(DD.max_dd)
#            rolling_dd_bnch = perf_Bench.rolling(window_length, min_periods=0).apply(DD.max_dd)

            rmdd_port = DD.rolling_max_dd(perf_Port.values, window_length, min_periods=1)
            rmdd_bnch = DD.rolling_max_dd(perf_Bench.values, window_length, min_periods=1)
            
            self.port_DD_label.setNum(np.min(rmdd_port))
            self.bck_DD_label.setNum(np.min(rmdd_bnch))
        
    def Universe_Selection(self):
        self.Portfolio_Benchmark.Name = self.UniverseSelect.currentText()
        self.Portfolio_Backtest.Name = self.UniverseSelect.currentText()

    def Add_To_Backtest_List(self):
        buffPort = port.Portfolio()
        
        buffPort.Dates = [self.startDate.dateTime().toPyDateTime(), 
                                         self.endDate.dateTime().toPyDateTime()]
        buffPort.Frequency = self.spinBox.value()
        buffPort.Histo_Length = self.spinBox_2.value() 
        buffPort.Wght_Constraint = [self.minW.value(), self.maxW.value()]
        buffPort.Wght_Histo = pd.DataFrame(index=np.unique(
                self.Portfolio_Benchmark.Historic_Composition.values[pd.notnull(
                        self.Portfolio_Benchmark.Historic_Composition.values)].flatten()))
        buffPort.Wght_Histo.index = buffPort.Wght_Histo.index + ' Equity'

        if self.radioButton.isChecked():
            buffPort.Opt_Strat = 'mv'
        elif self.radioButton_2.isChecked():
            buffPort.Opt_Strat = 'ew'
        elif self.radioButton_3.isChecked():
            buffPort.Opt_Strat = 'rb_beta'
        elif self.radioButton_4.isChecked():
            buffPort.Opt_Strat = 'rb_sigma'
        
        buffPort.Benchmark.__dict__ = dict(self.Portfolio_Benchmark.__dict__)
        
        self.Portfolio_List.append(port.Portfolio())
        self.Portfolio_List[len(self.Portfolio_List) - 1].__dict__ = dict(buffPort.__dict__)
        
        print("Portfolio Added")
    
    def Import_Portfolio(self):
        buffPort = self.Unpack_Pickle()
        
        self.Portfolio_List.append(port.Portfolio())
        self.Portfolio_List[len(self.Portfolio_List) - 1].__dict__ = dict(buffPort.__dict__)
        
        #self.Update_PortfolioSelect_List()
        print("Portfolio Added")
    
    def Refresh_Backtest_Context(self):
        # Reset the aggregated datas
        self.Bench_Strategy = pd.DataFrame()
        Bench_Index = pd.DataFrame()
        # Aggregate backtested Portfolios
        i=1
        benchNames = []
        for portBck in self.Portfolio_List:
            self.Bench_Strategy.loc[:,"%s_%d"%(portBck.Opt_Strat,i)] = pd.Series(portBck.Perf, index=portBck.Perf.index)
            if portBck.Benchmark.Name not in benchNames:
                benchNames.append(portBck.Benchmark.Name)
                bnch_Extr = portBck.Benchmark.Extract_Index_Returns([self.Bench_Strategy.index.min(),
                                                                     self.Bench_Strategy.index.max()])
                print(type(bnch_Extr))
                Bench_Index.loc[:, portBck.Benchmark.Name] = pd.Series(bnch_Extr.iloc[:,0], index=bnch_Extr.index)
            i+=1
        self.Bench_Strategy = pd.merge(self.Bench_Strategy, 
                                       Bench_Index, how='outer', left_index=True, right_index=True).fillna(0.0)
        
        self.Update_PortfolioSelect_List()
        
    
    def Reset_Backtest_List(self):
        self.Portfolio_List = []
        print("Portfolio List Flushed")
        self.Portfolio_Select.clear()
    
    def launcher_On_CLick(self):
        #QMessageBox.information(self, "Parameters","Universe : %s at index: %d" %(self.UniverseSelect.currentText(),self.UniverseSelect.currentIndex()))
        thread_list = []
        time1 = time.time()
        '''
        Portfolio Initialisation
        
        - Faire une fonction qui vérifie que la liste n'est pas vide
        - Pouvoir modifier l'historique d'optimisation pour la liste en cour
        - Faire une optimisation de la vue en fonction de la longueur
        '''      
#        print("Back Testing on Universe : %s on : %s" %(self.UniverseSelect.currentText(),self.startDate.date().toString(parserD)))
#        print("Constraint: %f / %f" %(self.minW.value(), self.maxW.value()))
        '''
        Launch Backtesting
        '''
        for portBck in self.Portfolio_List:
            if portBck.Opt_Strat == 'mv':
                thread_list.append(mp.Process(target=portBck.BackTest_MinVar()))
            elif portBck.Opt_Strat == 'ew':
                thread_list.append(mp.Process(target=portBck.BackTest_EqualWeight()))
            elif portBck.Opt_Strat == 'rb_beta':
                thread_list.append(mp.Process(target=portBck.BackTest_1_Beta()))
            elif portBck.Opt_Strat == 'rb_sigma':
                thread_list.append(mp.Process(target=portBck.BackTest_1_Sigma()))

        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

        time2 = time.time()
        print 'Computing take: \t\t- %0.5f ms' % ((time2-time1)*1000.0)
        '''
        Compile results, rebase and plot
        '''
        self.Refresh_Backtest_Context()
        self.Bench_Strategy = pd.merge(self.Bench_Strategy, 
                                       self.Portfolio_Benchmark.Extract_Index_Returns([self.Bench_Strategy.index.min(),
                                                                                       self.Bench_Strategy.index.max()]), how='inner', left_index=True, right_index=True)
        
        self.Bench_Strategy.fillna(0, inplace=True)
        self.Plot(100.0 * np.cumprod(1.0 + self.Bench_Strategy, axis=0))
        '''
        Update the list in the Risk Analysis tab
        '''
        self.Update_PortfolioSelect_List()
        
    def import_Historic_Data(self):
        self.Portfolio_Benchmark.Import_Data(self.Unpack_Pickle())
    
    def import_Historic_Compo(self):
        self.Portfolio_Benchmark.Historic_Composition = self.Unpack_Pickle()
    
    def import_Historic_Index(self):
        self.Portfolio_Benchmark.Historic_Index = self.Unpack_Pickle()
    
    def Unpack_Pickle(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            return pickle.load(open(fileName, 'rb'))
        else:
            return None
    
    def Save_as_Pickle(self):
        if len(self.Portfolio_List) > 0:
            i=1
            for portBck in self.Portfolio_List:
                pickle.dump(portBck, open('.//AppData//%s_%d.pickle'%(portBck.Opt_Strat,i), 'wb'))
                print('%s_%d portfolio saved as .//AppData//%s_%d.pickle'%(portBck.Opt_Strat,i,portBck.Opt_Strat,i))
                i+=1
        print('=========================== Backtesting saved as pickle files')
    def Periods_Diff(self, Datetime1, Datetime2):
        return (Datetime2.year-Datetime1.year)*12+(Datetime2.month-Datetime1.month)
    
    def Plot(self, DF):
        self.mpl.canvas.ax.clear()
        
        color_funds = th.mplColorMapSetting(np.linspace(0, 0.5, DF.shape[1]+1))
#        # (DF.shape[1] - 1) portfolios + Benchmark
#        color_peers = plt.cm.winter(np.linspace(0, 0.5, DF.shape[1] - 5 + 1))
#        # All Portfolios - (4 backtested + Benchmark) + 1
#        color_bench = plt.cm.ocean(np.linspace(0, 0.5, 2))
        
        for i in xrange(DF.shape[1]-1):
            self.mpl.canvas.ax.plot(DF.iloc[:,i], color=color_funds[i], label=DF.columns[i], alpha=1.0, linewidth=1.0)
        
        self.mpl.canvas.ax.plot(DF.iloc[:,i+1], color='#37C8FF', label=DF.columns[i+1], alpha=1.0, linewidth=1.8)
#        ax1.plot(MinVar_Strategies_Rebase.iloc[:,4], color=color_bench[1], label=MinVar_Strategies.columns[i+1], alpha=0.7, linewidth=2.0)
#        
#        for i in range(5,MinVar_Strategies_Rebase.shape[1]):
#            ax1.plot(MinVar_Strategies_Rebase.iloc[:,i], color=color_peers[i-4], label=MinVar_Strategies.columns[i], alpha=1.0, linewidth=0.5, linestyle='--')
        
        self.mpl.canvas.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        self.mpl.canvas.ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=self.Portfolio_Backtest.Frequency))
        self.mpl.canvas.ax.set_xlim(DF.index.min(), DF.index.max())
        self.mpl.canvas.ax.grid(which='minor', alpha=0.5)
        self.mpl.canvas.ax.grid(which='major', alpha=0.9)
        self.mpl.canvas.ax.legend()
        
        self.mpl.canvas.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        self.mpl.canvas.fig.autofmt_xdate()
        self.mpl.canvas.fig.tight_layout()
        self.mpl.canvas.draw()
    
    def Refresh_plot(self):
        self.Plot(100.0 * np.cumprod(1.0 + self.Bench_Strategy[self.Graph_startDate.dateTime().toPyDateTime() : self.Graph_endDate.dateTime().toPyDateTime()], axis=0))

    def Plot_Perf_Analysis(self):
        if self.Portfolio_Select.count() > 0:
            self.mpl_returns.canvas.ax1.clear()
            self.mpl_returns.canvas.ax2.clear()
            self.mpl_returns.canvas.ax3.clear()
            
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.max()
            idx_Min = self.startDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.min()
            
            window_length = self.spinBox_drawdown.value()
            
            color_line = th.mplColorMapSetting(np.linspace(0, 0.8, 4))
            
            perf_Port = (np.cumprod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]], axis=0) - 1.0) * 100.0
            perf_Bench = (np.cumprod(1.0 + self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index], axis=0) - 1.0) * 100.0
            DD_Port = perf_Port.rolling(window_length, min_periods=0).apply(DD.max_dd)
            overPerf = perf_Port - perf_Bench
            
            self.mpl_returns.canvas.ax1.plot(perf_Bench, color='#37C8FF', linestyle='--')
            self.mpl_returns.canvas.ax1.plot(perf_Port, color=color_line[1])
            
            self.mpl_returns.canvas.ax2.plot(overPerf, color='#A9A9A9')
            d = overPerf.index.values 
            self.mpl_returns.canvas.ax2.fill_between(d, y1=overPerf, y2=0, where=overPerf >= 0, facecolor='green', alpha=0.3, interpolate=True)
            self.mpl_returns.canvas.ax2.fill_between(d, y1=overPerf, y2=0, where=overPerf <= 0, facecolor='red', alpha=0.3, interpolate=True)
            self.mpl_returns.canvas.ax3.plot(DD_Port, color=color_line[3], label='Rolling Drawdown %d' % window_length)

            self.mpl_returns.canvas.ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
            self.mpl_returns.canvas.ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            self.mpl_returns.canvas.ax1.set_xlim(idx_Min, idx_Max)
            self.mpl_returns.canvas.ax1.grid(which='minor', alpha=0.5)
            self.mpl_returns.canvas.ax1.grid(which='major', alpha=0.9)
            self.mpl_returns.canvas.ax1.legend()
            self.mpl_returns.canvas.ax2.legend()
            self.mpl_returns.canvas.ax3.legend()
            
            self.mpl_returns.canvas.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            self.mpl_returns.canvas.fig.autofmt_xdate()
            self.mpl_returns.canvas.fig.tight_layout()
            self.mpl_returns.canvas.draw()
    
    def Plot_Volatility(self):
        if self.Portfolio_Select.count() > 0:
            self.mpl_vol.canvas.ax.clear()
            
            port_Idx = self.Portfolio_Select.currentIndex()
            col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.max()
            idx_Min = self.startDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.min()
            
            window = self.spinBox_volatility.value()
            
            p_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]]
            b_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index]            
            
            p_rolling_Vol = p_returns.rolling(window).std() * np.sqrt(250)
            b_rolling_Vol = b_returns.rolling(window).std() * np.sqrt(250)
            p_rolling_Vol.dropna(inplace=True)
            b_rolling_Vol.dropna(inplace=True)
            
            self.mpl_vol.canvas.ax.plot(b_rolling_Vol, color='#37C8FF', linestyle='--')
            self.mpl_vol.canvas.ax.plot(p_rolling_Vol)
            
            self.mpl_vol.canvas.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
            self.mpl_vol.canvas.ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            self.mpl_vol.canvas.ax.set_xlim(idx_Min, idx_Max)
            self.mpl_vol.canvas.ax.grid(which='minor', alpha=0.5)
            self.mpl_vol.canvas.ax.grid(which='major', alpha=0.9)
            self.mpl_vol.canvas.ax.legend()
            
            self.mpl_vol.canvas.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            self.mpl_vol.canvas.fig.autofmt_xdate()
            self.mpl_vol.canvas.fig.tight_layout()
            self.mpl_vol.canvas.draw()

    def Plot_Distribution(self):
        if self.Portfolio_Select.count() > 0:
            self.mpl_distrib.canvas.ax.clear()
            port_Idx = self.Portfolio_Select.currentIndex()
            #col_Name_Index = self.Bench_Strategy.columns[self.Bench_Strategy.shape[1]-1]
            idx_Max = self.endDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.max()
            idx_Min = self.startDate_2.dateTime().toPyDateTime() #self.Bench_Strategy.index.min()

            p_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, self.Bench_Strategy.columns[port_Idx]]
            #b_returns = self.Bench_Strategy.loc[idx_Min:idx_Max, col_Name_Index]
            
            #Edges, Bins = oph.Opt_Histogram((p_returns - np.mean(p_returns))/ np.std(p_returns),200)
            #Nb_bins = oph.Opt_nb_bins((p_returns - np.mean(p_returns))/ np.std(p_returns))
            Nb_bins = 11
            
            # Plot Histogram
            bins, Edges, _ = self.mpl_distrib.canvas.ax.hist(p_returns, Nb_bins-1, normed=True, facecolor='green', alpha=0.75)
            # add a 'best fit' line
            y = mlab.normpdf(Edges, np.mean(p_returns), np.std(p_returns))
            l = self.mpl_distrib.canvas.ax.plot(Edges, y, 'r--', linewidth=1)
            
            self.mpl_distrib.canvas.fig.tight_layout()
            self.mpl_distrib.canvas.draw()
            
    def Plot_Turnover(self):
        if self.Portfolio_Select_2.count() > 0:
            self.mpl_2.canvas.ax.clear()
            port_Idx = self.Portfolio_Select_2.currentIndex()
            idx_Max = self.endDate_3.dateTime().toPyDateTime() 
            idx_Min = self.startDate_3.dateTime().toPyDateTime() 
            
            compo_histo = self.Portfolio_List[port_Idx].Wght_Histo.loc[:,idx_Min:idx_Max]
            turnover = np.abs(compo_histo.diff(axis=1)).sum(axis=0)*100.0
            index = np.arange(len(turnover))
            self.mpl_2.canvas.ax.bar(index, turnover.values)
            
            #self.mpl_2.canvas.ax.xaxis.set_label_text(turnover.index.to_datetime(), rotation=25)  
            #self.mpl_2.canvas.ax.legend(.index, mode='expand', ncol=6)
            #self.mpl_2.canvas.fig.autofmt_xdate()
            #self.mpl_2.canvas.fig.tight_layout()
            self.mpl_2.canvas.draw()
            
    def Plot_Sectorial_Repartition(self):
        if self.Portfolio_Select_2.count() > 0:
            self.mpl_3.canvas.ax.clear() 
            port_Idx = self.Portfolio_Select_2.currentIndex()
            idx_Max = self.endDate_3.dateTime().toPyDateTime() 
            idx_Min = self.startDate_3.dateTime().toPyDateTime() 

            compo_histo = self.Portfolio_List[port_Idx].Wght_Histo.loc[:,idx_Min:idx_Max]
            expo = pd.merge(compo_histo, self.ref_eqty, 
                            left_index=True, right_index=True).fillna(0.0).groupby('GICS_SECTOR_NAME').sum()
            
            # color bar
            colors = th.mplColorMapSetting(np.linspace(0, 1, len(expo) + 1))
            # offset to bar stacked 
            y_offset = np.zeros(len(expo.columns))
            # locations
            index = np.arange(len(expo.columns))
            
            for row in xrange(len(expo)):
                self.mpl_3.canvas.ax.bar(index, 
                                         expo.iloc[row,:].values, 
                                         width=0.5, bottom=y_offset, color=colors[row])
                y_offset += expo.iloc[row,:].values
            
            self.mpl_3.canvas.ax.xaxis.set_label_text(expo.columns.to_datetime(), rotation=25)  
            self.mpl_3.canvas.ax.legend(expo.index, mode='expand', ncol=6)
            #self.mpl_3.canvas.fig.autofmt_xdate()
            #self.mpl_3.canvas.fig.tight_layout()
            self.mpl_3.canvas.draw()
            
    def Plot_Country_Repartition(self):
        if self.Portfolio_Select_2.count() > 0:
            self.mpl_4.canvas.ax.clear()        
            port_Idx = self.Portfolio_Select_2.currentIndex()
            idx_Max = self.endDate_3.dateTime().toPyDateTime() 
            idx_Min = self.startDate_3.dateTime().toPyDateTime() 

            compo_histo = self.Portfolio_List[port_Idx].Wght_Histo.loc[:,idx_Min:idx_Max]
            expo = pd.merge(compo_histo, self.ref_eqty, 
                            left_index=True, right_index=True).fillna(0.0).groupby('COUNTRY_FULL_NAME').sum()
            
            # color bar
            colors = th.mplColorMapSetting(np.linspace(0, 1, len(expo) + 1))
            # offset to bar stacked 
            y_offset = np.zeros(len(expo.columns))
            # locations
            index = np.arange(len(expo.columns))
            
            for row in xrange(len(expo)):
                self.mpl_4.canvas.ax.bar(index, 
                                         expo.iloc[row,:].values, 
                                         width=0.5, bottom=y_offset, color=colors[row])
                y_offset += expo.iloc[row,:].values
            
            self.mpl_4.canvas.ax.xaxis.set_label_text(expo.columns.to_datetime(), rotation=25)  
            self.mpl_4.canvas.ax.legend(expo.index, mode='expand', ncol=6)
            #self.mpl_4.canvas.fig.autofmt_xdate()
            #self.mpl_4.canvas.fig.tight_layout()
            self.mpl_4.canvas.draw()
            
def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()