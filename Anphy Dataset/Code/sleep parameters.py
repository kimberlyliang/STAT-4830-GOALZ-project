# -*- coding: utf-8 -*-

"""
anphy
"""


#### Calculate the box plot

import plotly.express as px
import pandas as pd
from IPython.display import HTML 
import plotly.io as pio
pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# read data
df = pd.read_csv('path')

TRT =go.Violin(
    y = df.iloc[:,3],  # data
    box_visible=True,  # invisible line display
    line_color='mediumblue',  # the color of line 
    meanline_visible=True,  # visible meanline display
    fillcolor='lightblue',  # fillcolor setting
    opacity=0.6,  # opacity
    x0='TRT',  # x axis title 
    points="all",
    name='TRT',

)

TST =go.Violin(
    y = df.iloc[:,4],  
    box_visible=True,  
    line_color='deeppink',  
    meanline_visible=True,  
    fillcolor='lightpink',  
    opacity=0.7,  
    x0='TST',   # 
    points="all",
    name='TST'
)

SOL =go.Violin(
    y = df.iloc[:,5],  
    box_visible=True,  
    line_color='purple',  
    meanline_visible=True,  
    fillcolor='mediumpurple', 
    opacity=0.7, 
    x0='SOL' ,
    points="all",
    name='SOL'
)
REML =go.Violin(
    y = df.iloc[:,6],  
    box_visible=True,  
    line_color='gray',  
    meanline_visible=True,  
    fillcolor='lightgray',  
    opacity=0.6,  
    x0='REML',   
    points="all" ,
    name='REML'
)

WASO =go.Violin(
    y = df.iloc[:,7],  
    box_visible=True,  
    line_color='green',  
    meanline_visible=True,  
    fillcolor='yellowgreen',  
    opacity=0.6,  
    x0='WASO' ,  
    points="all",
    name='WASO'
)


N1 =go.Violin(
    y = df.iloc[:,8],  
    box_visible=True,  
    line_color='black',  
    meanline_visible=True,  
    fillcolor='lightseagreen',  
    opacity=0.6,  
    x0='N1',   
    points="all" ,
    name='N1'
)
N2 =go.Violin(
    y = df.iloc[:,9],  
    box_visible=True,  
    line_color='green',  
    meanline_visible=True,  
    fillcolor='lightgreen',  
    opacity=0.6,  
    x0='N2',   #
    points="all" ,
    name='N2'
)

N3 =go.Violin(
    y = df.iloc[:,10],  
    box_visible=True,  
    line_color='Red',  
    meanline_visible=True,  
    fillcolor='rosybrown',  
    opacity=0.6,  
    x0='N3',   
    points="all" ,
    name='N3'
)

REM =go.Violin(
    y = df.iloc[:,11],  
    box_visible=True,  
    line_color='yellowgreen',  
    meanline_visible=True,  
    fillcolor='green',  
    opacity=0.6,  
    x0='REM',   
    points="all" ,
    name='REM'
)

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
# set plotly Default theme
pio.templates.default = 'plotly_white'
data = [TRT,TST,SOL,REML,WASO,N1,N2,N3,REM]


fig = go.Figure(data = data )




# set plotly Default theme
pio.templates.default = 'plotly_white'

# 
pd.set_option('display.max_columns', None)

fig.update_layout(
	font_size=16,
    plot_bgcolor='white',
    width=1000,
    margin=dict(t=100,pad=10)
)

# Add X and Y axis labels
fig.update_xaxes(title_text='Sleep parameters')
fig.update_yaxes(title_text='Durations(min)')
fig.update_layout(
    title={
        'text': "Sleep macrostructure of healthy subjects at group level",   # tltle
        'y':0.9, 
        'x':0.5,
        'xanchor': 'center',   # location
        'yanchor': 'top'})
fig.update_layout(legend_font_size=14)

fig.show()
# set plotly Default theme
pio.templates.default = 'plotly_white'


# Import package, offline draw figures 
import plotly
#Use init_notebook_mode() to view the plots in jupyter notebook
plotly.offline.init_notebook_mode()


