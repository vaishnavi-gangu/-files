#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:18:29 2019

@author: student
"""

import numpy as np

def merge(input_list):

    i1 = np.vstack((input_list[0][0],input_list[0][1],input_list[0][2],input_list[0][3],input_list[0][4],input_list[0][5],input_list[0][6],input_list[0][7],input_list[0][8],input_list[0][9],input_list[0][10],input_list[0][11],input_list[0][12],input_list[0][13],input_list[0][14],input_list[0][15]))
    i2 = np.vstack((input_list[0][16],input_list[0][17],input_list[1][0],input_list[1][1],input_list[1][2],input_list[1][3],input_list[1][4],input_list[1][5],input_list[1][6],input_list[1][7],input_list[1][8],input_list[1][9],input_list[1][10],input_list[1][11],input_list[1][12],input_list[1][13]))
    i3 = np.vstack((input_list[1][14],input_list[1][15],input_list[1][16],input_list[1][17],input_list[2][0],input_list[2][1],input_list[2][2],input_list[2][3],input_list[2][4],input_list[2][5],input_list[2][6],input_list[2][7],input_list[2][8],input_list[2][9],input_list[2][10],input_list[2][11]))
    i4 = np.vstack((input_list[2][12],input_list[2][13],input_list[2][14],input_list[2][15],input_list[2][16],input_list[2][17],input_list[3][0],input_list[3][1],input_list[3][2],input_list[3][3],input_list[3][4],input_list[3][5],input_list[3][6],input_list[3][7],input_list[3][8],input_list[3][9]))
    i5 = np.vstack((input_list[3][10],input_list[3][11],input_list[3][12],input_list[3][13],input_list[3][14],input_list[3][15],input_list[3][16],input_list[3][17],input_list[4][0],input_list[4][1],input_list[4][2],input_list[4][3],input_list[4][4],input_list[4][5],input_list[4][6],input_list[4][7]))
    i6 = np.vstack((input_list[4][8],input_list[4][9],input_list[4][10],input_list[4][11],input_list[4][12],input_list[4][13],input_list[4][14],input_list[4][15],input_list[4][16],input_list[4][17],input_list[5][0],input_list[5][1],input_list[5][2],input_list[5][3],input_list[5][4],input_list[5][5]))
    i7 = np.vstack((input_list[5][6],input_list[5][7],input_list[5][8],input_list[5][9],input_list[5][10],input_list[5][11],input_list[5][12],input_list[5][13],input_list[5][14],input_list[5][15],input_list[5][16],input_list[5][17],input_list[6][0],input_list[6][1],input_list[6][2],input_list[6][3]))
    i8 = np.vstack((input_list[6][4],input_list[6][5],input_list[6][6],input_list[6][7],input_list[6][8],input_list[6][9],input_list[6][10],input_list[6][11],input_list[6][12],input_list[6][13],input_list[6][14],input_list[6][15],input_list[6][16],input_list[6][17],input_list[7][0],input_list[7][1]))
    i9 = np.vstack((input_list[7][2],input_list[7][3],input_list[7][4],input_list[7][5],input_list[7][6],input_list[7][7],input_list[7][8],input_list[7][9],input_list[7][10],input_list[7][11],input_list[7][12],input_list[7][13],input_list[7][14],input_list[7][15],input_list[7][16],input_list[7][17]))
    i10 = np.vstack((input_list[8][0],input_list[8][1],input_list[8][2],input_list[8][3],input_list[8][4],input_list[8][5],input_list[8][6],input_list[8][7],input_list[8][8],input_list[8][9],input_list[8][10],input_list[8][11],input_list[8][12],input_list[8][13],input_list[8][14],input_list[8][15]))
    i11 = np.vstack((input_list[8][16],input_list[8][17],input_list[9][0],input_list[9][1],input_list[9][2],input_list[9][3],input_list[9][4],input_list[9][5],input_list[9][6],input_list[9][7],input_list[9][8],input_list[9][9],input_list[9][10],input_list[9][11],input_list[9][12],input_list[9][13]))
    i12 = np.vstack((input_list[9][14],input_list[9][15],input_list[9][16],input_list[9][17],input_list[10][0],input_list[10][1],input_list[10][2],input_list[10][3],input_list[10][4],input_list[10][5],input_list[10][6],input_list[10][7],input_list[10][8],input_list[10][9],input_list[10][10],input_list[10][11]))
    i13 = np.vstack((input_list[10][12],input_list[10][13],input_list[10][14],input_list[10][15],input_list[10][16],input_list[10][17],input_list[11][0],input_list[11][1],input_list[11][2],input_list[11][3],input_list[11][4],input_list[11][5],input_list[11][6],input_list[11][7],input_list[11][8],input_list[11][9]))
    i14 = np.vstack((input_list[11][10],input_list[11][11],input_list[11][12],input_list[11][13],input_list[11][14],input_list[11][15],input_list[11][16],input_list[11][17],input_list[12][0],input_list[12][1],input_list[12][2],input_list[12][3],input_list[12][4],input_list[12][5],input_list[12][6],input_list[12][7]))
    i15 = np.vstack((input_list[12][8],input_list[12][9],input_list[12][10],input_list[12][11],input_list[12][12],input_list[12][13],input_list[12][14],input_list[12][15],input_list[12][16],input_list[12][17],input_list[13][0],input_list[13][1],input_list[13][2],input_list[13][3],input_list[13][4],input_list[13][5]))
    i16 = np.vstack((input_list[13][6],input_list[13][7],input_list[13][8],input_list[13][9],input_list[13][10],input_list[13][11],input_list[13][12],input_list[13][13],input_list[13][14],input_list[13][15],input_list[13][16],input_list[13][17],input_list[14][0],input_list[14][1],input_list[14][2],input_list[14][3]))
    
    input_data = np.hstack((i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16))
    
    return input_data