

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


import bokehelect as electrostatics
import numpy
from bokehelect import (ElectricField, GaussianCircle, PointCharge,
                            Potential)


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
charges = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xs = np.zeros(len(charges))
ys = np.zeros(len(charges))
nlines = 4
source = ColumnDataSource(dict(q=charges,x=xs,y=ys))
num_charges = len(charges)
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
    _q = convert_column(source.data['q'])
    _x = convert_column(source.data['x'])
    _y = convert_column(source.data['y'])
   
    charges = []
    for  k in range(len(_x)):
        charges.append(PointCharge(_q[k], [_x[k], _y[k]]))

    field = ElectricField(charges)
    potential = Potential(charges)

    r = np.array([_x,_y]).T
    tree = cKDTree(r)

    fieldlines = []
    
    for k in range(len(charges)):
        if _q[k] == 0 :
            continue
        # find closest charge
        d,ii = tree.query(r,k=1)
        idx = np.where(ii>0)[0][0]
        dr = r[idx]-r[k]
        
        angle = np.arctan(dr[1]/dr[0])
        print(k,angle,dr)
        g = GaussianCircle(charges[k].x, 0.05,angle)
        for fp in g.fluxpoints(field,nlines,uniform=True):
            fieldlines.append(field.line(fp))

    x, y = np.meshgrid(
            np.linspace(XMIN, XMAX,128),
            np.linspace(XMIN, XMAX,128))
      
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    z[i, j] = logmodulus(potential.magnitude([x[i, j], y[i, j]]))

    new_contour_data = contour_data(x, y, z, levels)
    contour_renderer.set_data(new_contour_data)

    xs = []
    ys= []
    starts,ends = [],[]
    for k,fieldline in enumerate(fieldlines):
        X,Y = zip(*fieldline.x)
        xs.append(X)
        ys.append(Y)
        n = int(len(X)/2) if len(X) < 225 else 75
        starts.append([X[n],Y[n]])
        ends.append([X[n+1],Y[n+1]])


    # update line data source
    line_sources.data = dict(
        xs= xs,
        ys= ys
    )
    starts = np.array(starts)
    ends = np.array(ends)
    arrow_source.data = dict(x_start=starts[:,0], x_end=ends[:,0],y_start=starts[:,1], y_end=ends[:,1])

    # update arrow source


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
            np.linspace(XMIN, XMAX,200),
            np.linspace(XMIN, XMAX,200))
z = np.zeros_like(x)
u = np.zeros_like(x)

for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = logmodulus(potential.magnitude([x[i, j], y[i, j]]))

levels = np.linspace(-2,2,9)

# contour_renderer = fig.image(image='z', source=contour_source,  palette="Sunset11")
contour_renderer = fig.contour( x, y, z, levels=levels, 
    fill_color=Plasma,
)

xs = []
ys= []
starts,ends = [],[]
for k,fieldline in enumerate(fieldlines):
    X,Y = zip(*fieldline.x)
    n = int(len(X)/2) if len(X) < 225 else 75
    starts.append([X[n],Y[n]])
    ends.append([X[n+1],Y[n+1]])
    xs.append(X)
    ys.append(Y)
    
# plot field lines
line_sources  = ColumnDataSource(dict(
        xs= xs,
        ys= ys
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