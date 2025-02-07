# -*- coding: utf-8 -*-
"""
Anphy 
"""

import yasa
import numpy as np
import matplotlib.pyplot as plt


 # plot the hypnogram
nd = np.genfromtxt('path name', delimiter=',', skip_header=True)
 
final_list = nd.tolist()

fig, ax = plt.subplots(1, 1, figsize=(7, 3), constrained_layout=True)
ax = yasa.plot_hypnogram(final_list, fill_color="lightblue", ax=ax)