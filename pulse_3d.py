import taichi as ti
import numpy as np

ti.init(arch=ti.cpu,advanced_optimization=True,fast_math=True)
N = 200
dt=ti.field(ti.f32,shape=())
dx=ti.field(ti.f32,shape=())
c=ti.field(ti.f32,shape=())
t=ti.field(ti.f32,shape=())
phi=ti.field(ti.f32,shape=())
A=ti.field(ti.f32,shape=())
omega=ti.field(ti.f32,shape=())
damping=ti.field(ti.f32,shape=())

dt[None] = 1*10**-6 #millisecond
dx[None] = 3 / N   
c[None]=299.72458  #m/μs [meter/microsecond]
phi[None]=0
omega[None]=5000 #rad/sec
A[None]=1.7 #amplitude
damping[None]=2500
max_z=ti.field(ti.f32,shape=())
min_z=ti.field(ti.f32,shape=())

v = ti.field(ti.f32, shape=((N+2) , (N+2)))
surface=ti.Vector.field(3,ti.f32,shape=(N + 2,N + 2))
sphere=ti.Vector.field(3,ti.f32,shape=(N + 2,N + 2))
pos=ti.Vector.field(3,ti.f32,shape=((N + 2) * (N + 2)))
pos2=ti.Vector.field(3,ti.f32,shape=((N + 2) * (N + 2)))
vertex_color=ti.Vector.field(3,ti.f32,shape=((N + 2) * (N + 2)))
@ti.kernel
def init():
    for i in range(N+2):
        for j in range(N+2):
            surface[i,j]=ti.Vector([i*dx[None],j*dx[None], 0])
            v[i,j]=0

    t[None]=0

@ti.kernel
def cast2sphere():
    k=0
    for i in range(N+2):
        for j in range(N+2):
            theta = i * np.pi / (N+1)
            phi = j * 2 * np.pi / (N+1)
            x = ti.math.sin(theta) * ti.math.cos(phi)
            y = ti.math.sin(theta) * ti.math.sin(phi)
            z = ti.math.cos(theta)
            pos2[k] = ti.Vector([x,y,z]) - surface[i,j][2]
            k+=1
@ti.kernel
def update():
    surface[N//2,N//2]=ti.Vector([N//2,N//2 ,A[None]*ti.sin(omega[None]*t[None]*ti.math.pi + phi[None])])
    k=0
    max_z[None]=0
    min_z[None]=0
    ti.loop_config(parallelize=8*8*2, block_dim=16*16*2)

    for i in range(1, N+1):
        for j in range(1, N+1):
            v[i,j] += c[None]**2 * ( (surface[i + 1, j][2] + surface[i - 1, j][2] - 2 * surface[i,j][2]) / dx[None]**2) * dt[None]
            v[i,j] += c[None]**2 * ( (surface[i, j + 1][2] + surface[i, j - 1][2] - 2 * surface[i,j][2]) / dx[None]**2) * dt[None]
            v[i,j] += c[None]**2 * ( (surface[i + 1, j + 1][2] + surface[i - 1, j - 1][2] - 2 * surface[i,j][2]) / dx[None]**2) * dt[None]
            v[i,j] += c[None]**2 * ( (surface[i + 1, j - 1][2] + surface[i - 1, j + 1][2] - 2 * surface[i,j][2]) / dx[None]**2) * dt[None]
            v[i,j] *= ti.exp(-damping[None] * dt[None])
            surface[i,j]+= ti.Vector([0, 0, v[i,j] * dt[None] ])
            if surface[i,j][2]>max_z[None]:
                max_z[None]=surface[i,j][2]
            if surface[i,j][2]<min_z[None]:
                min_z[None]=surface[i,j][2]
            k+=1
    t[None]+=dt[None]
    k=0
    ti.loop_config(parallelize=8*8*2, block_dim=16*16*2)
    for i,j in ti.ndrange(N+2,N+2):
        surface[0, j] = ti.Vector([0,j,0])#surface[1, j]#
        surface[N+1, j] = ti.Vector([N+1,j,0])#surface[N, j]
        surface[i, 0] = ti.Vector([i,0,0])#surface[i, 1]
        surface[i, N+1] = ti.Vector([i,N+1,0])#surface[i, N]
        pos[k]=surface[j,i]
        pos2[k]=surface[j,i+1]
        r=0
        b=0
        if pos[k][2]>=max_z[None]*0.01: 
            r= 1
        if pos[k][2]<=min_z[None]*0.01:
            b= ti.exp(-pos[k][2]*10)
        else:
            b=0
        g=  0.5 if r==0 else 0.5
        vertex_color[k]=ti.Vector([r,g, b])
        k+=1

init()
window = ti.ui.Window('Window Title', res = (1000, 1000), pos = (150, 150))
canvas = window.get_canvas()
#canvas.set_background_color((1,1,1))
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

camera.up(0,0,1)
x,y,z=4.6,4,1.78
lx,ly,lz=0,0,-2
camera.lookat(lx,ly,lz)
camera.position(x,y,z)
scene.set_camera(camera)
scene.ambient_light([1,1,1])
n=1
rad=0.005
new_color=(147/255, 235/255, 150/255)
sphere_flag=False

scene.ambient_light((1,1,1))
while window.running:
    sphere_flag=gui.checkbox("cast to sphere",sphere_flag)
    rad=gui.slider_float("rad", rad, 0.0001, 0.01)
    x=gui.slider_float("x",x , -10, 100)
    y=gui.slider_float("y",y , -10, 100)
    z=gui.slider_float("z",z , -3, 30)
    lx=gui.slider_float("lx",lx , -10, 10)
    ly=gui.slider_float("ly",ly , -10, 10)
    lz=gui.slider_float("lz",lz , -10, 10)
    n = gui.slider_float("i", n, 1, 100)
    omega[None] = gui.slider_float("freq", omega[None], 4000, 10000)
    A[None] = gui.slider_float("Amp", A[None], 0, 10)
    phi[None] = gui.slider_float("phi", phi[None], 0, 180)
    damping[None] = gui.slider_float("damp", damping[None], 0, 20000)
    #c[None]=gui.slider_float("c", c[None], 1, 5000)
    reset=gui.button("reset")
    if reset: init()

    camera.position(x,y,z)
    camera.lookat(lx,ly,lz)
    new_color = gui.color_edit_3("color", new_color)
    scene.set_camera(camera)
    canvas.scene(scene)
    
    for i in range(int(n)):
        update()
        if sphere_flag: cast2sphere()
        scene.particles(pos2, rad, per_vertex_color=vertex_color)
    window.show()



