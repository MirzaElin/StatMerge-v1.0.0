import math
def _mean(x): x=list(x); return sum(float(v) for v in x)/len(x)
def _var(x,ddof=1): x=list(x); m=_mean(x); return sum((float(v)-m)**2 for v in x)/(len(x)-ddof)
def cohen_d(x,y):
    x=list(x); y=list(y); nx,ny=len(x),len(y)
    mx,my=_mean(x),_mean(y); vx,vy=_var(x),_var(y)
    sp=((nx-1)*vx+(ny-1)*vy)/(nx+ny-2)
    return (mx-my)/math.sqrt(sp)
def hedges_g(x,y):
    d=cohen_d(x,y); n=len(list(x))+len(list(y)); J=1-(3/(4*n-9)); return J*d
