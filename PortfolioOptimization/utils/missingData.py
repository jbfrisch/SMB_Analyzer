# -*- coding: utf-8 -*-

ms_data_list = ['APAM NA Equity',
                '63DU GY Equity',
                'FI IM Equity',
                'DB11 GY Equity', 
                'NXPI UW Equity',
                'CCE NA Equity',
                'ATCT NA Equity',
                'IGY GY Equity',
                'EVK GY Equity',
                'ABN NA Equity',
                'FCA IM Equity',
                'CNHI IM Equity',
                'SFR FP Equity',
                'BKIA SQ Equity',
                'ATC NA Equity',
                'ATCB NA Equity',
                'IAG SQ Equity',
                'AENA SQ Equity',
                'AMS SQ Equity']


def rm_data(ticker, d_frame):
    if ticker in d_frame.columns:
        d_frame.drop(ticker, axis=1, inplace=True)

def check_ms_data_OLD(tk_list, d_frame):
    for ticker in ms_data_list : rm_data(ticker, d_frame)
    
def check_ms_data(d_frame):
    ms_data = d_frame.isnull().values.any(axis=0)
    d_frame.drop(d_frame.columns[ms_data.tolist()], inplace=True)