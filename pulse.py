import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)
N = 50000

dt=ti.field(ti.f32,shape=())
dx=ti.field(ti.f32,shape=())
c=ti.field(ti.f32,shape=())
t=ti.field(ti.f32,shape=())
damping=ti.field(ti.f32,shape=())

dt[None] = 1e-6 #microsecond
dx[None] = 1 / N  
c[None]=299.792458 #meter / microsecond
damping[None]=0
x = ti.field(ti.f32, shape=N + 2)  
y = ti.field(ti.f32, shape=N + 2)  
v = ti.field(ti.f32, shape=N + 2)  
pos=ti.Vector.field(2,ti.f32,shape=N)
pos2=ti.Vector.field(2,ti.f32,shape=N)
pos3=ti.Vector.field(2,ti.f32,shape=N)


@ti.kernel
def init():
    for i in range(N + 2):
        x[i] = (i - 1) * dx[None]
        y[i] = 0.5 #+ ( ( ti.math.sin(60*i/N *ti.math.pi)-ti.math.sin(30*i/N *ti.math.pi) ) / N *ti.math.pi )
        v[i] = 0
        pos[i]=ti.Vector([x[i],y[i]])
    t[None]=0



@ti.kernel
def update():

    if t[None]<dt[None]*100:
        y[N//2]=0.5
        #y[0]=0.8
        #y[N//2]=0.5*ti.math.sin(t[None])
    elif t[None]>dt[None]*100 and t[None]<dt[None]*200:
        y[N//2]=0.7
        #y[N//2]=0.2*ti.math.sin(t[None]*np.pi*omega) + 0.5
        #y[0]=0.2
        #y[0]= ( ( ti.math.sin(2*t[None]*ti.math.pi *omega)-ti.math.sin(t[None]*ti.math.pi*omega) ) / ti.math.pi*t[None]*omega )*0.01 +0.5
    elif t[None]==dt[None]*200: #and t[None]<dt[None]*3003:
        #y[N//2]=0.2*ti.math.sin(t[None]*np.pi*omega) + 0.5
        y[N//2]=0.5
    
    for i in range(1, N):
        #v[i] += c[None]**2 * ( (y[i + 1] + y[i - 1] - 2 * y[i]) / dx[None]**2) * dt[None]
        v[i] += c[None]**2 * ( (y[i + 1] + y[i - 1] - 2 * y[i]) / dx[None]**2) * dt[None]
        v[i] *= ti.exp(-damping[None] * dt[None])
        y[i] += v[i] * dt[None]
    
    
    for i in range(N):
        pos[i]=ti.Vector([x[i],y[i]])
        pos2[i]=ti.Vector([x[i],y[i]])
        pos3[i]=ti.Vector([x[i],y[i]])
    y[0] = y[1]
    y[N+1]=y[N]
    t[None]+=dt[None]

init()

window = ti.ui.Window('Window Title', res = (3440, 1000), pos = (150, 150),vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((0,0,0))
n=1
rad=0.001
new_color=(147/255, 235/255, 150/255)
dtmul=1
while window.running:
    for i in range(int(n)):
        update()
    n = gui.slider_float("i", n, 1, 100)
    damping[None] = gui.slider_float("damp", damping[None], 0, 2000)
    

    dtmul= gui.slider_float("dt", dtmul, 1, 10)
    dt[None]=dtmul*1e-8
    reverse=gui.button("reverse")
    val=gui.slider_float("c", c[None], 1, 500)
    reset=gui.button("reset")
    c[None] = val if not reverse else -val
    if reset: init()
    rad=gui.slider_float("rad", rad, 0.0001, 0.01)
    new_color = gui.color_edit_3("color", new_color)

    canvas.circles(pos,rad,color=(new_color))
    canvas.circles(pos2,rad,color=(new_color))
    canvas.circles(pos3,rad,color=(new_color))
    window.show()


