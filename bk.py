from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Plasma256 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row,column
from bokeh.models import Arrow, VeeHead, Select, DataTable,TableColumn,ColumnDataSource,MultiLine
from bokeh.plotting.contour import contour_data
from bokeh.models import Button, Div
import numpy as np

from scipy.spatial.distance import pdist
import bokehelect as electrostatics
from bokehelect import (ElectricField, GaussianCircle, PointCharge,
                            Potential)

# import warnings
# warnings.filterwarnings('error')

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

def nlines(charge):
    return 4+int(abs(charge)/0.1)
def min_diff_pos(array_like, target):
    return np.abs(np.array(array_like)-target).argmin()
def logmodulus(x):
    return np.sign(x)*(np.log10(np.abs(x)+1))


XMIN, XMAX =-1, 1
YMIN, YMAX = -1, 1
ZOOM = 1
XOFFSET = 0.0
nlevels=18
electrostatics.init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)
levels = np.linspace(-2,2,nlevels)
many_levels = np.linspace(-2,2,256)

# create table of charges
inputs = []
qs = [1]
_x = np.zeros(len(qs))
_y = np.zeros(len(qs))  

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
    update()

def update():
    "Update plot"
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

    r = np.array([_x,_y]).T
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
            closest =-1
            for p in range(len(_x)):
                if p!=k and _q[p]!=0 :
                    dr = r[p]-r[k]
                    dr2 = (dr**2).sum()
                    if dr2<_dr2:
                        _dr = dr
                        _dr2 = _dr2
                        closest = p

            # if _q[k]<_q[closest]:
                # _dr*=-1
            angle = np.arctan2(_dr[1],_dr[0])
            # print(k,angle,dr)
            g = GaussianCircle(charges[k].x, 0.05,angle)
            for fp in g.fluxpoints(field,nlines(_q[k])):
                fieldlines.append(field.line(fp))

    else:
        g = GaussianCircle(charges[0].x, 0.05)
        for fp in g.fluxpoints(field,nlines(_q[0])):
            fieldlines.append(field.line(fp))
    
    # evaluate the potential
    z = coulomb_pot(r,x,y,_q)
    # transform it to improve the visualisation
    z = logmodulus(z)
    levels = np.linspace(z.min(),z.max(),nlevels)
    many_levels = np.linspace(-2,2,256)
    print(z.min(), z.max())
    new_contour_data = contour_data(x, y, z, levels, want_fill=False)
    contour_renderer.set_data(new_contour_data)
    many_contour_data = contour_data(x, y, z, many_levels, want_line=False)
    field_renderer.set_data(many_contour_data)


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

def on_change_selection(attrname, old,new):
    if new == 'single':
        _q = [1.0]
        _x = [0.0]
        _y = [0.0]
    if new == 'dipole':
        _q = [1.0,-1]
        _x = [-0.5,0.5]
        _y = [0.0,0]
    if new == 'triangle':
        R = 0.8
        theta = np.pi/2
        _q = [1.0,-1.0,1.0]
        _x = [R*np.cos(theta),R*np.cos(theta+2*np.pi/3),R*np.cos(theta+4*np.pi/3)]
        _y = [R*np.sin(theta),R*np.sin(theta+2*np.pi/3),R*np.sin(theta+4*np.pi/3)]
    if new == 'square':
        _q = [1.0,-1,1.0,-1.0]
        _x = [-0.5,0.5,0.5,-0.5]
        _y = [-0.5,-0.5,0.5,0.5]
    if new == 'random':
        n = 8
        _q = np.random.uniform(0.05,1,n)*np.random.choice([-1,1],n)
        # _x = np.random.uniform(-1,1,n)
        _x = [ np.random.uniform(-1,1)]
        _y =[ np.random.uniform(-1,1)]
        while len(_y)<n:
            xtrial = np.random.uniform(-1,1)
            ytrial = np.random.uniform(-1,1)
            _x.append(xtrial)
            _y.append(ytrial)
            r=np.array([_x,_y]).T
            print(r)
            if  min(pdist(r))<0.1:
                _x.pop()
                _y.pop()
            else:
                pass
                
    if new == 'empty':
        n = 15
        _q = np.zeros(n)
        _x = np.zeros(n)
        _y = np.zeros(n)  
    
    source.data = dict(q=np.round(_q,2),x=np.round(_x,2),y=np.round(_y,2))
    # update()


##########################################


# Set up the charges, electric field, and potential
charges = [PointCharge(1, [0, 0]),
        #    PointCharge(-1, [0.5, 0])
           
           ]
field = ElectricField(charges)
potential = Potential(charges)

# Set up one Gaussian surface
g = GaussianCircle(charges[0].x, 0.1)
# Create the field lines
fieldlines = []
for fp in g.fluxpoints(field,8):
    fieldlines.append(field.line(fp))
# fieldlines.append(field.line([10, 0]))

x, y = np.meshgrid(
            np.linspace(XMIN, XMAX,512),
            np.linspace(XMIN, XMAX,512))

z = coulomb_pot(np.array([_x,_y]).T,x,y,qs)
z = logmodulus(z)

field_renderer = fig.contour( x, y, z, levels=many_levels, 
    fill_color=Plasma256, 
    # line_color='black',
    # line_width=0
    )
contour_renderer = fig.contour( x, y, z, levels=levels, 
    # fill_color=Plasma256, 
    line_color='black',#['#75d5ff']*int(nlevels/2) + ['black']*int(nlevels/2),
    line_dash=['solid']*int(nlevels/2) + ['dashed']*int(nlevels/2),
    line_width=1.)

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

selection = Select(title="Configuration:", value="single", options=["single", "dipole", "triangle", "square","random","empty"])
selection.on_change('value',on_change_selection)
compute = Button(label="Compute", button_type="primary",align='end')
compute.on_click(on_click_callback)

title =  Div(text='<h1 style="text-align: center">Electric Field and Potential of Multiple Charges</h1>\n<h2>   @<a href="https://francescoturci.net" target="_blank">Francesco Turci</a> </h2>')
layout = column(title,row(fig, column(row(selection,compute),data_table)) )    
# Div(text='by <a href="https://francescoturci.net" target="_blank"> Francesco Turci</a>') )
curdoc().title = "Electric Field and Potential of Multiple Charges"
curdoc().add_root(layout)