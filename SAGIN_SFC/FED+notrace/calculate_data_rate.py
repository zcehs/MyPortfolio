import sys
import os
import math
import satellite_locate as sat_locate 
from SAGIN_topology_big import (
    node_s,
    node_a,
    node_g,
    node_location
)

AD_TRANSMITTER_GAIN = [10 ** 3.28 for x in range(node_s)] + [10 ** 2.7 for x in range(node_a)] + [10 ** 4.32 for x in range(node_g)]
AD_RECEIVER_GAIN = [10 ** 3.28 for x in range(node_s)] + [10 ** 2.3 for x in range(node_a)] + [10 ** 4.32 for x in range(node_g)]
TRANSMIT_POWER = [100 for x in range(node_s)] + [10 ** 0.9 for x in range(node_a)] + [10 for x in range(node_g)]#W

FREQUENCY_BAND = 2.8e10  # 28 GHz

C = 3e8  # Speed of light in m/s

K = 1.38e-23  # Boltzmann constant in J/K

T = 300  # Standard temperature in Kelvin #300K

B = 2e8  # Bandwidth in Hz # 10MHz = 1e7Hz





def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)

def get_distance(reciever, transmitter, time):
    pos1 = node_location[reciever]
    pos2 = node_location[transmitter]
    if reciever < node_s:
        pos1 = sat_locate.Satellite_Locate(time)
    elif transmitter < node_s:
        pos2 = sat_locate.Satellite_Locate(time)
    """print("trans:", pos2)
    print("recv:", pos1)
    print("distance:", calculate_distance(pos1, pos2))"""
    return calculate_distance(pos1, pos2)

def FSPL(distance):
    """Calculate Free Space Path Loss (FSPL) in dB."""
    if distance == 0:
        return 0
    
    fspl = C / (4 * math.pi * distance * FREQUENCY_BAND)
    fspl = math.pow(fspl, 2)
    return fspl

def Channel_Gain(reciever, transmitter, time):
    h = AD_TRANSMITTER_GAIN[transmitter] * AD_RECEIVER_GAIN[reciever] * FSPL(get_distance(reciever, transmitter, time))
    return h

def SNR(reciever, transmitter, time):
    h = Channel_Gain(reciever, transmitter, time)
    snr = h * TRANSMIT_POWER[transmitter] / (K * T * B)
    return snr

def Data_Rate(reciever, transmitter, time):
    if reciever >= (node_s + node_a) and transmitter >= (node_s + node_a):
        return 10 ** 10
    snr = SNR(reciever, transmitter, time)
    rate = B * math.log2(1 + snr)
    return rate

'''print(Data_Rate(7,8,0))
print(Data_Rate(7,8,10))
print(Data_Rate(6,1,10))
print(Data_Rate(1,2,0))
print(Data_Rate(7,0,0))
print(Data_Rate(7,0,10))
print(Data_Rate(7,0,15))'''
