from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Plasma, Accent 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row,column
from bokeh.models import Arrow, VeeHead, Slider, DataTable,TableColumn,ColumnDataSource,MultiLine
from bokeh.plotting.contour import contour_data
from bokeh.models import Button, Div
from scipy.spatial import cKDTree
import numpy as np

from joblib import Parallel, delayed

import bokehelect as electrostatics
import numpy
from bokehelect import (ElectricField, GaussianCircle, PointCharge,
                            Potential)

import warnings
warnings.filterwarnings('error')
# import numba as nb
# @nb.njit
def coulomb_pot(r,xgrid,ygrid, charges):
    dx = r[0,0]-xgrid
    dy = r[0,1]-ygrid
    dr = np.sqrt(dx**2+dy**2)
    invdr = charges[0]/dr
    for p in range(1,r.shape[0]):
        dx = r[p,0]-xgrid
        dy = r[p,1]-ygrid
        dr = np.sqrt(dx**2+dy**2)
        invdr += charges[p]/dr
    return invdr

def min_diff_pos(array_like, target):
    return np.abs(np.array(array_like)-target).argmin()
def logmodulus(x):
    return np.sign(x)*(np.log10(np.abs(x)+1))
# pylint: disable=invalid-name

XMIN, XMAX =-1, 1
YMIN, YMAX = -1, 1
ZOOM = 1
XOFFSET = 0.0

electrostatics.init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)


# create table of charges


inputs = []
qs = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
_x = np.zeros(len(qs))
_y = np.zeros(len(qs))
nlines = 4
source = ColumnDataSource(dict(q=qs,x=_x,y=_y))
columns = [TableColumn(field="q",title="Charge"),TableColumn(field="x",title="x"),TableColumn(field="y",title="y")]
data_table = DataTable(columns=columns, source=source,editable=True)
fig = figure(
    width=600, #pixels
    height=600,
    x_range=(XMIN, XMAX),
    y_range=(YMIN, YMAX)
    )

fig.toolbar.logo = None
fig.toolbar_location = None
fig.xgrid.visible = False
fig.ygrid.visible = False


def convert_column(col):
    v = []
    for value in col:
        try:
            v.append(float(value))
        except:
            v.append(0) 
    return v           

# def on_change_callback(attr,old,new):
def on_click_callback():
    global z
    _q = convert_column(source.data['q'])
    _x = convert_column(source.data['x'])
    _y = convert_column(source.data['y'])
   
    charges = []
    for  k in range(len(_x)):
        if _q[k]!=0:
            charges.append(PointCharge(_q[k], [_x[k], _y[k]]))

    field = ElectricField(charges)
    potential = Potential(charges)

    print("here")
    r = np.array([_x,_y]).T
    print(r)

    fieldlines = []
    
    print(np.nonzero(_q))
    print(_q)
    if len(np.nonzero(_q)[0])>1:
        for k in range(len(charges)):
            if _q[k] == 0 :
                continue
            # find closest charge
            closest = -1
            _dr2 = 10.0
            _dr = 0
            for p in range(len(_x)):
                if p!=k and _q[p]!=0 :
                    dr = r[p]-r[k]
                    dr2 = (dr**2).sum()
                    if dr2<_dr2:
                        _dr = dr
                        _dr2 = _dr2

            
            angle = np.arctan(_dr[1]/_dr[0])
            # print(k,angle,dr)
            g = GaussianCircle(charges[k].x, 0.05,angle)
            for fp in g.fluxpoints(field,nlines):
                fieldlines.append(field.line(fp))

    else:
        g = GaussianCircle(charges[k].x, 0.05)
        for fp in g.fluxpoints(field,nlines):
            fieldlines.append(field.line(fp))


    # def evaluate_pot(i,j):
    #     return potential.magnitude([x[i, j], y[i, j]])

    # for i in range(x.shape[0]):
    #     z[i] = Parallel(n_jobs=4)(delayed(evaluate_pot)(i,j) for j in range(x.shape[1]))
    # z = logmodulus(z)
    z = coulomb_pot(r,x,y,_q)
    z = logmodulus(z)

    new_contour_data = contour_data(x, y, z, levels)
    contour_renderer.set_data(new_contour_data)

    xl = []
    yl= []
    starts,ends = [],[]
    for k,fieldline in enumerate(fieldlines):
        X,Y = zip(*fieldline.x)
        xl.append(X)
        yl.append(Y)
        n = int(len(X)/2) if len(X) < 225 else 75
        starts.append([X[n],Y[n]])
        ends.append([X[n+1],Y[n+1]])


    # update line data source
    line_sources.data = dict(
        xs= xl,
        ys= yl
    )

    # update arrow source
    starts = np.array(starts)
    ends = np.array(ends)
    arrow_source.data = dict(x_start=starts[:,0], x_end=ends[:,0],y_start=starts[:,1], y_end=ends[:,1])



# data_table.source.on_change('data', on_change_callback)
# Set up the charges, electric field, and potential
charges = [PointCharge(1, [0, 0]),
        #    PointCharge(-1, [0.5, 0])
           
           ]
field = ElectricField(charges)
potential = Potential(charges)

# Set up the Gaussian surface
g = GaussianCircle(charges[0].x, 0.1)

# Create the field lines
fieldlines = []
for fp in g.fluxpoints(field,nlines):
    fieldlines.append(field.line(fp))
# fieldlines.append(field.line([10, 0]))

x, y = np.meshgrid(
            np.linspace(XMIN, XMAX,128),
            np.linspace(XMIN, XMAX,128))
# z = np.zeros_like(x)

z = coulomb_pot(np.array([_x,_y]).T,x,y,qs)
z = logmodulus(z)
# def evaluate_pot(i,j):
#     return potential.magnitude([x[i, j], y[i, j]])


# for i in range(x.shape[0]):
#     z[i] = Parallel()(delayed(evaluate_pot)(i,j) for j in range(x.shape[1]))


levels = np.linspace(-2,2,9)

# contour_renderer = fig.image(image='z', source=contour_source,  palette="Sunset11")
contour_renderer = fig.contour( x, y, z, levels=levels, 
    fill_color=Plasma,
)

xl = []
yl= []
starts,ends = [],[]
for k,fieldline in enumerate(fieldlines):
    X,Y = zip(*fieldline.x)
    n = int(len(X)/2) if len(X) < 225 else 75
    starts.append([X[n],Y[n]])
    ends.append([X[n+1],Y[n+1]])
    xl.append(X)
    yl.append(Y)
    
# plot field lines
line_sources  = ColumnDataSource(dict(
        xs= xl,
        ys= yl
    )
)

starts = np.array(starts)
ends = np.array(ends)
arrow_source = ColumnDataSource(dict(x_start=starts[:,0], x_end=ends[:,0],y_start=starts[:,1], y_end=ends[:,1]))


glyph = MultiLine(xs='xs', ys='ys',line_color='white')
fig.add_glyph(line_sources, glyph)
vh = VeeHead(size=7, fill_color='white',line_color='white')
fig.add_layout(Arrow(end=vh, x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end', source=arrow_source))


button = Button(label="Compute", button_type="primary")
button.on_click(on_click_callback)

title =  Div(text='<h1 style="text-align: center">Potential and Field Lines of Multiple Point Charges</h1>\n by <a href="https://francescoturci.net" target="_blank"> Francesco Turci</a>')
layout = column(title,row(fig, column(data_table,button)), )    
# Div(text='by <a href="https://francescoturci.net" target="_blank"> Francesco Turci</a>') )
curdoc().title = "Potential and Field Lines of Multiple Point Charges"
curdoc().add_root(layout)