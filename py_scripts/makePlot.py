
# coding: utf-8

# In[252]:

# make plot of gauge, radar, percentile, rose, motion box, motion line & cpt dot


# In[253]:
import cairocffi
import numpy as np
import pandas as pd
import sys
from matplotlib import cm, rc
from matplotlib import pyplot as plt
# sys.path.insert(0,'../utilities/')
from py_scripts.utilities.loadData import db_manager
from py_scripts.utilities.mathHelper import smooth
from windrose import WindroseAxes
from pygal.style import LightenStyle, LightColorizedStyle,DefaultStyle
import pygal
from matplotlib.patches import Circle, Wedge, Rectangle
from IPython.display import display


# In[254]:



def drawMotionIndex(motionIndex,cpt_results,path,showPlot):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.linspace(0,13,len(motionIndex)),motionIndex,color=theme_color)
    ax.set_yticks(np.linspace(0,8,5))
    ax.set_xticks(np.linspace(0,14,8))
    ax.set_xlim(0,13)
    ax.set_ylim(0,8)
    cpt_color = [standard_color3[2],standard_color3[1],standard_color3[0]] # correct, commission, omission
    for hit,c in zip(cpt_results,cpt_color):
        ax.scatter(hit,[4]*len(hit),color=c)
    ax.set_xlabel('Minute',fontsize=22)
    ax.set_ylabel('Motion Index',fontsize=22)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(color666)
    ax.spines['bottom'].set_color(color666)
#     ax.yaxis.label.set_color(color666)
#     ax.xaxis.label.set_color(color666)
#     ax.tick_params(axis='x', colors=color666)
#     ax.tick_params(axis='y', colors=color666)
    
    plt.legend(['How much the child \nmoved during the test'],loc='upper right')#,bbox_to_anchor=(1.25, 0.7))
#     plt.tight_layout(pad = 6)
    plt.grid()
    if showPlot:
        plt.show()
    fig.savefig(path+'motionIndex.png', dpi=200,box_inches=25)
    
# drawMotionIndex(rd1.motion_index,rd1.cpt_response_min)


# In[255]:

def drawMotionBox(pts,path,showPlot):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(-pts[:,0],pts[:,1],color=theme_color)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.set_xlabel('Left and Right')
    ax.set_ylabel('Front and Back')
    ax.set_yticks(np.linspace(-0.4,0.4,5))
    ax.set_xticks(np.linspace(-0.4,0.4,5))

    rect = Rectangle((-0.2,-0.2), 0.4, 0.4, facecolor='none', edgecolor=color999, ls = ':')
    ax.add_patch(rect)
    ax.set_aspect(1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color(color666)
#     ax.spines['bottom'].set_color(color666)
#     ax.yaxis.label.set_color(color666)
#     ax.xaxis.label.set_color(color666)
#     ax.tick_params(axis='x', colors=color666)
#     ax.tick_params(axis='y', colors=color666)
    
    if showPlot:
        plt.show()
    fig.savefig(path+'motionBox.png', dpi=200,bbox_inches='tight')
    
# drawMotionBox(rd1.motion_box)


# In[256]:

# https://pypi.python.org/pypi/windrose
# %debug
from windrose import WindroseAxes
import numpy as np
from matplotlib import pyplot as plt
def drawRose(value,path,showPlot):
    wd = [10]*360
    ws = np.arange(360)
    ws=[1]*360

    #     fig,_ = plt.subplots(figsize=(8,8))
    ax,fig = WindroseAxes.from_ax()
#     ax.bar(wd, ws, normed=True, nsector=360, bins=np.array([1]), opening=0.8,colors=theme_color,bin_value=value)

    ax.bar(wd, ws, normed=True, nsector=360,bins=1, opening=0.8,colors=theme_color,bin_value=value)
    ax.set_xticklabels(['','Door','White Board','','Window']+['']*4)
    fig.savefig(path+'Rose.png', dpi=200,bbox_inches='tight')
    if showPlot:
        plt.show()

# drawRose(rd1.roseValue)


# In[257]:

# http://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/

# %matplotlib inline

import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

from matplotlib.patches import Circle, Wedge, Rectangle


def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], colors='jet_r', cat=1, title='', score = 50, fname='./meter.png'): 
    
    """
    some sanity checks first
    
    """
        
    color666 = '#333333'
    
    N = len(labels)
    
    if cat > N: 
        raise Exception("\n\nThe category ({}) is greated than the length\nof the labels ({})".format(cat, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots(figsize = (8,4))

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .5, *ang, facecolor='w', lw=2, alpha=0))
        # arcs
        patches.append(Wedge((0.,0.), .5, *ang, width=0.10, facecolor=c, lw=2, alpha=1))
    
    [ax.add_patch(p) for p in patches]

    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.435 * np.cos(np.radians(mid)), 0.435 * np.sin(np.radians(mid)), lab,             horizontalalignment='center', verticalalignment='center', fontsize=14,             rotation = rot_text(mid),color=color666)


    
    ax.text(0, -0.1, title, horizontalalignment='center',          verticalalignment='center', fontsize=25,color=color666)
#     ax.text(0, -0.4, '60', horizontalalignment='center', \
#          verticalalignment='center', fontsize=28,color=color666)

    """
    plots the arrow now
    """
    
#     pos = mid_points[abs(cat - N)]
    pos = 180 - score/100*180

    ax.arrow(0, 0, 0.35 * np.cos(np.radians(pos)), 0.35 * np.sin(np.radians(pos)),                  width=0.01, head_width=0.03, head_length=0.05, fc=color666, ec=color666)
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor=color666))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    fig.savefig(fname, dpi=200)

def drawGauge(score,path,showPlot):
    gauge(labels=['LOW','MEDIUM','HIGH'], colors=standard_color3, cat=3,           title='Overall Score: %d'%score, score = score, fname=path+'gauge.png') 


# gauge(labels=['LOW','MEDIUM','HIGH'], colors=['#F77B28','#FFCE3D','#00A854'], cat=3, title='Overall Score: 60', fname='./gauge.png') 
# drawGauge(10)


# In[258]:

# https://matplotlib.org/examples/api/radar_chart.html
# %debug
"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
	
		num_vars : int
			Number of variables for radar chart.
		frame : {'circle' | 'polygon'}
			Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), radius = 0.5, color='r')

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                self._draw_point(line)

        def _draw_point(self,line):
            x, y = line.get_data()
            super().plot(x,y,'b.',markersize=10,clip_on=False)
            
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels,frac = 1.15, fontsize=14)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def drawRadar(title,case_data,path,showPlot):
    N = len(title)
    theta = radar_factory(N, frame='circle')

#     data = example_data()
    
    title = title[0:1]+title[-1:0:-1]
    case_data = case_data[0:1]+case_data[-1:0:-1]
    
    spoke_labels = title


    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    color = theme_color
    # Plot the four cases from the example data on separate axes
    ax.set_rmin(0)
    ax.set_rmax(1)
    ax.set_ylim((0,1))
    lines,labels = ax.set_rgrids([50,100],angle=90, color = 'grey')
#     ax.set_grid()

    if max(case_data) < 1:
        case_data = [c*100 for c in case_data]

    ax.plot(theta, case_data, color=color)
    ax.fill(theta, case_data, facecolor=color, alpha=0.5)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
#     ax = axes[0, 0]
#     labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
#     legend = ax.legend(labels, loc=(0.9, .95),
#                        labelspacing=0.1, fontsize='small')

#     fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
#              horizontalalignment='center', color='black', weight='bold',
#              size='large')

    if showPlot:
        plt.show()
    fig.savefig(path+'Radar.png', dpi=200)
# %debug


# In[259]:

# http://www.pygal.org/en/stable/documentation/configuration/label.html#y-labels-major
# http://www.pygal.org/en/stable/_modules/pygal/graph/radar.html?highlight=radar
# http://www.pygal.org/en/stable/documentation/configuration/rendering.html#stroke
import pygal
# config = pygal.config
# config.fill = True
# %debug
def drawRadar_pygal(title,case_data,path,showPlot):
    title = title[0:1]+title[-1:0:-1]
    case_data = case_data[0:1]+case_data[-1:0:-1]
    
    dark_lighten_style = LightenStyle(theme_color,step=5, max_=10)
    dark_lighten_style1 = LightenStyle('#ff0000')
    colorset = ['#cfefdf','#a7e1c4','#76d0a3']
    
    dark_lighten_style.opacity = 0.5
    dark_lighten_style.background = '#ffffff'
    dark_lighten_style.font_family = "DejaVu Sans"
    dark_lighten_style.legend_font_family = "DejaVu Sans"
    dark_lighten_style.major_label_font_family = "DejaVu Sans"
    dark_lighten_style.title_font_family = "DejaVu Sans"
    dark_lighten_style.tooltip_font_family = "DejaVu Sans"
    dark_lighten_style.label_font_family = "DejaVu Sans"
    dark_lighten_style.label_font_size = 40
    dark_lighten_style.major_label_font_size = 40
    dark_lighten_style.legend_font_size = 40
    dark_lighten_style.colors=[colorset[1],theme_color]
#     dark_lighten_style.foreground = 'rgba(0, 0, 0, .87)'
#     raise

    radar_chart = pygal.Radar(show_legend=True,width=1600,height=1200,style=dark_lighten_style,
                              margin=25, spacing=20, stroke_style={'width': 5},dots_size=8,show_dots=1,stroke=1)
    # radar_chart.title = 'V8 benchmark results'
    radar_chart.x_labels = title
    if max(case_data) < 1:
        case_data = [c*100 for c in case_data]
#     radar_chart.add('Chrome1', [{'value':50,'style': 'fill: False; stroke: red; stroke-width: 4;stroke-dasharray: 15, 10, 5, 10, 15'}]*len(case_data))
    radar_chart.add('Standard score', [50]*len(case_data),fill = True,show_dots = 0, stroke_style={'width': 2, 'dasharray': '3, 6'})

    radar_chart.add('Your score', case_data)
#     radar_chart._fill('red')
    radar_chart.y_labels = [0,50,100]

#     radar_chart.render_to_file('plot/Radar_pygal.svg',fill = True)
    radar_chart.render_to_png(path+'Radar_pygal.png',fill = True)
    if showPlot:
        display({'image/svg+xml': radar_chart.render(fill = True)}, raw=True)


# In[260]:


from pygal import Config
from pygal.style import LightenStyle, LightColorizedStyle

# percentile = rd1.percentile
# x_label = rd1.percentileName
# percentile = [0.1,0.5,0.9]
# x_label = ['0']

def selectColor(score):
    if score > 2/3:
        return standard_color3[0]
    elif score > 1/3:
        return  standard_color3[1]
    else:
        return standard_color3[2]

def drawPercentile(percentile,x_label,path,showPlot):

    config = Config()
    config.show_legend = False
    # config.range = (0,1)

    dark_lighten_style = LightenStyle('#336676', base_style=LightColorizedStyle)
    dark_lighten_style.background = '#ffffff'
    dark_lighten_style.opacity=1
    dark_lighten_style.font_family = "DejaVu Sans"
    dark_lighten_style.legend_font_family = "DejaVu Sans"
    dark_lighten_style.major_label_font_family = "DejaVu Sans"
    dark_lighten_style.title_font_family = "DejaVu Sans"
    dark_lighten_style.tooltip_font_family = "DejaVu Sans"
    dark_lighten_style.label_font_family = "DejaVu Sans"
    dark_lighten_style.label_font_size = 40
    dark_lighten_style.major_label_font_size = 40

#     print(dark_lighten_style.to_dict())

    bar_chart = pygal.Bar(config,width=1600,height=1000,rounded_bars=6, style=dark_lighten_style, margin=0)

    bar_chart.x_labels = x_label#['Concentration', 'Stability', 'Focus Continuity'] #map(str, range(2002, 2013))
    bar_chart.y_labels = (0, 0.2, 0.4, 0.6, 0.8, 1)

    # bar_chart.add('Percentile0', rd1.percentile[0])
    # bar_chart.add('Percentile1', rd1.percentile[1])
    # bar_chart.add('Percentile2', rd1.percentile[2])


    bar_chart.add('Percentile', [
     {'value': 1- percentile[0],'color' : selectColor(percentile[0])},
     {'value': 1- percentile[1], 'color': selectColor(percentile[1])},
     {'value': 1-percentile[2], 'color': selectColor(percentile[2])}
    ])
    if showPlot:
        display({'image/svg+xml': bar_chart.render()}, raw=True, dpi=200)
    
    bar_chart.render_to_png(path+'Percentile.png',fill = True, dpi=200)
# drawPercentile(percentile,x_label)


# In[261]:

# class to pack data necessary to produce report
class reportData(object):

    # initialize necessary data field
    def __init__(self):
        self.caseid = 0
        self.bayes_score = 50 # init value of bayes score
        
        self.numRadarElem = 6 # number of elements in radar plot
        self.radar = [50]*self.numRadarElem # init value of radar plot
        self.radarName = ['Attention','Self Control','Speed','Sensitivity','Cautiousness','Steadiness']
        
        self.percentileElem = 3 # number of elements in percentile plot
        self.percentile = [50]*self.percentileElem # init value of percentile plot
        self.percentileName = ['Concentration','Stability','Focus Continuity']
        
        self.motion_box_numpoints = 3900 # not working at a fixed length now
        self.motion_box = np.zeros((self.motion_box_numpoints,2))
        self.motion_index_points = 390 # not working at a fixed length now
        self.motion_index = [0]*self.motion_index_points
        
        self.roseBinNum = 360
        self.roseValue = [0]*self.roseBinNum
        
        self.cpt_response_min = [np.arange(52)*1.5*10/60,[],[]] # list0 for correct, list1 for commmission, list2 for omission
        
        self.db = db_manager()
        
    def loadFromDb(self,caseid,result_db = 'vrclassroomdata'):
        self.caseid = caseid
        self.loadTableFromDb(caseid,result_db)
        
        self.bayes_score = 100*(1-self.bayes_table['FinalProbabilityOfADHD'].iloc[0])
        self.loadFromDb_Radar()
        self.loadFromDb_Percentile()
        self.loadFromDb_MotionBox()
        self.loadFromDb_MotionIndex()
        self.loadFromDb_TrialData()
        self.loadFromDb_Rose()
        
    
    def loadTableFromDb(self,caseid,result_db):
        self.db.connect(result_db)
        
        self.bayes_table = self.db.fetch_table('bayes_probabilities',['*'],'caseid',caseid)
        self.head_features_table = self.db.fetch_table('head_features',['*'],'caseid',caseid)
        self.hmd_data_table = self.db.fetch_table('hmd_data',['*'],'caseid',caseid)
        self.trial_data_table = self.db.fetch_table(table_name = 'trial_data',field_names = ['*'],                                              where_clause='where BlockNumber>0 and caseid=%d'%caseid)
        self.rose_table = self.db.fetch_table('head_rot',['*'],'caseid',caseid)
        self.percentile_table = self.db.fetch_table('percentile',['*'],'caseid',caseid)
        print('load table done')
        self.db.close()
        
        
    def loadFromDb_Radar(self):
        """ change Beta to C"""
        fieldNames_BayesRadar = ['OmissionRawProbabilityHealthy',                     'CommissionRawProbabilityHealthy',                     'TargetRTVRawProbabilityHealthy',                     'DPrimeRawProbabilityHealthy',                     'CRawProbabilityHealthy']
        for i in range(len(fieldNames_BayesRadar)):
            self.radar[i] = self.bayes_table[fieldNames_BayesRadar[i]].iloc[0]
        self.radar[-1] = 1- self.head_features_table['TimeActive'].iloc[0]
    
    def loadFromDb_Rose(self):
        self.roseValue = self.rose_table['Value'].tolist()
        
    def loadFromDb_Percentile(self):
        """ read from percentile instead"""
        fieldNames_Percentile = ['PerTimeActive',                                'PerPathLen',                                'PerNumRot']
        self.percentile = [self.percentile_table[fieldNames_Percentile[i]].iloc[0]                           for i in range(len(fieldNames_Percentile))]
    
    def loadFromDb_MotionBox(self):
        pos_xz = np.asarray(self.hmd_data_table[['PosX','PosZ']])
        pos_xz = np.subtract(pos_xz,pos_xz[0,:])
        self.motion_box = pos_xz[::10,:]

        
    def smooth_MotionIndex(self,speed):
        speed = speed[::10]
        speed = smooth(y=speed,box_pts=300)
        speed = speed[::10]
        return speed*5000
        
    def loadFromDb_MotionIndex(self):
        pos_xyz = self.hmd_data_table[['PosX','PosY','PosZ']]
        diff_xyz = np.diff(pos_xyz,axis=0)
        speed = np.sqrt(np.sum(diff_xyz**2,axis=1))
        self.motion_index = self.smooth_MotionIndex(speed)
        
    def loadFromDb_TrialData(self):
        isTarget = self.trial_data_table['IsTarget']
        isRight = self.trial_data_table['RightOrWrong']
        correct_TrialId = np.asarray(self.trial_data_table[(isTarget==1) & (isRight==1)]['TrialId'])
        commission_TrialId = np.asarray(self.trial_data_table[(isTarget==0) & (isRight==0)]['TrialId'])
        omission_TrialId = np.asarray(self.trial_data_table[(isTarget==1) & (isRight==0)]['TrialId'])
        self.cpt_response_min = [correct_TrialId*1.5/60, commission_TrialId*1.5/60, omission_TrialId*1.5/60]
        
    def loadFromJson(self,caseid,json_str):
        return
    
    def showPlot(self,path='plot/',showPlot=True):
        drawGauge(score=self.bayes_score,path=path,showPlot=showPlot)
#         drawRadar(case_data=self.radar,title=self.radarName)
        drawRadar_pygal(case_data=self.radar,title=self.radarName,path=path,showPlot=showPlot)
        drawRose(value = self.roseValue,path=path,showPlot=showPlot)
        drawMotionBox(pts=self.motion_box,path=path,showPlot=showPlot)
        drawPercentile(percentile=self.percentile,x_label=self.percentileName,path=path,showPlot=showPlot)
        drawMotionIndex(cpt_results=self.cpt_response_min,motionIndex=self.motion_index,path=path,showPlot=showPlot)
    
        return


# In[268]:

theme_color = '#2696D3'
standard_color3 = ['#F77B28','#FFBF00','#00A854'] # sad, warning, good
color666 = '#666666'
color999 = '#999999'
font = {'family' : 'serif' ,
        'weight' : 100,
        'size'   : 20,
       'style'   : 'normal'}
rc('font', **font)

import sys
import getopt

def main(caseid,path='plot/',showPlot=True, db_name = 'webtest'):
    rd1 = reportData()
    # rd1.loadFromDb(21)

    rd1.loadFromDb(caseid,result_db=db_name)

    rd1.showPlot(path,showPlot)

if __name__=='__main__':
    try:
        options,args = getopt.getopt(sys.argv[1:],"hc:p:d:",["help","caseid=","path=","dbname="])
    except getopt.GetoptError:
        print('Usage: makePlot.exe -c/--caseid caseid -p/--path path -d/--db db_name')
        raise Exception(2)
        
    CaseId = 50
    path = 'plot/'
    showPlot = True
    db_name = 'vrclassroomdata'
    for name,value in options:
        if name in ("-h","--help"):
            print("""Usage: makePlot.exe -c caseids -p path -d db_name \n\nDefault:    \n\tcaseids: 50 \n\tpath: plot/ \n\tdb_name: vrclassroomdata""")
            sys.exit(1)

        if name in ("-c","--caseid"):
            try:
                CaseId = int(value)
            except Exception as e:
                raise Exception (2)
            showPlot = False
                
        if name in ("-p","--path"):
            path = value
            showPlot=False
            
        if name in ("-d","--dbname"):
            db_name = value
            showPlot=False
            
    main(CaseId,path,showPlot,db_name)


# In[263]:

# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

