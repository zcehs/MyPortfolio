import numpy as np
import scipy.special
import random
import time
import os
import copy
import math
import numpy as np

ini_eva_angle = np.deg2rad(45) #初始仰角(轉弧度)
r_earth = 6378137 #地球半徑(m)
h_sat = 500000 #衛星高度(m)
r_sat = r_earth + h_sat #衛星軌道半徑 (m)
ini_earth_angle = np.pi/2 - np.arcsin(r_earth/r_sat * np.sin(np.pi/2 + ini_eva_angle)) - ini_eva_angle #求初始地心角，正弦定理
rad_spd_sat = np.sqrt(6.67e-11 * 5.97*10e24/ r_sat**3) #衛星角速度 (rad/s)

def Satellite_Locate(t):
	angle_change = t*rad_spd_sat % (2*ini_earth_angle) #仰角限制
	earth_angle = ini_earth_angle - angle_change #計算當前地心角
	d = np.sqrt(r_earth**2 + r_sat**2 - 2*r_earth*r_sat*np.cos(np.abs(earth_angle))) #計算衛星與座標(0,0)之距離
	eva_angle = np.arccos(r_sat/d * np.sin(np.abs(earth_angle))) #計算仰角，利用sin(90度+theta) = cos(theta)
	if earth_angle < 0: eva_angle = np.pi - eva_angle	
	x = d * np.cos(eva_angle)
	z = d * np.sin(eva_angle)
	return [x, 0, z]

# 測試用
# for i in range(100):
# 	print(Satellite_Locate(i))