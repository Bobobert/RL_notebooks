"""
    All importations, functions, and constants for main use in here,

"""
# Importations here
import numpy as np
import math
from numba import njit
from numba.typed import List

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import ray
import os

import cv2
import matplotlib.pyplot as plt 

from gym import make

# Functions
ceil = math.ceil
Tsum = torch.sum
Tsqrt = torch.sqrt
Tlog = torch.log


# Consts
NCPUS = os.cpu_count()
F_DTYPE_DEFT = torch.float32
DEVICE_DEFT = torch.device("cpu")

config = {
        "n-step": 10,
        "gamma" : 0.99,
        "learningRate" : 1e-5,
        "optimizer" : "Adam",
        "optimizerArgs" : {},
        "nActions" : 18,
        "lHist" : 4,
        "atari": True,
        "entropyLoss": 0.001,
        "nTest" : 20,
        "stepsTest" : -1,
        "nAgents" : 1,
        "cPolicyLoss" : 1.0,
        "cValueLoss": 0.6,
        "gradClip": 1e3,
        "episodes_train" : 10**3,
        "freq_test" : 100,
    }
