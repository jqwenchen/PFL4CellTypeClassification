#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w, num_samples):
    num_all_samples = np.sum(num_samples)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= num_samples[0] / num_all_samples
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * (num_samples[i] / num_all_samples)

    return w_avg
