# -*- coding: utf-8 -*-
"""
anphy
"""

import matplotlib.pyplot as plt
import pandas as pd


#plot the age distribution
df = pd.read_csv('path')

df = pd.read_csv('path')
gender=[16,18]
plt.pie(gender,startangle = 180,counterclock = False,)
print(df.iloc[:,3])
plt.title("gender")
plt.show()


## historm of age by gender
import seaborn as sns
# Histograms for each gender
df = pd.read_csv('path')
sns.distplot(a=df[df.Sex=='Female']['Age'], label="Female", kde=True)
sns.distplot(a=df[df.Sex=='Male']['Age'], label="Male", kde=True)

# Add title
plt.title("Histogram of age, by sex")
plt.grid (False)
# Force legend to appear
plt.legend()