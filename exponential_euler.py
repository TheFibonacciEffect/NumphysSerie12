# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.linalg import expm
# %%
def expo_euler_step(yk, h, f, J):
    print(J, J.shape)
    def phi(z):
        # return (expm(z) - 1) @ np.linalg.inv(z)
        return (expm(z) - np.eye(2)) @ np.linalg.inv(z)
        # return expm1_v(z)/z
        # return np.expm1(z)/z
    print(np.shape(h*phi(h*J) @ f(*yk)), np.shape(f(*yk)))
    return yk + h*phi(h*J) @ f(*yk)

def wrapper(f, Df, N, y0, t):
    t, h = np.linspace(0,t, N, retstep=True)
    y = np.zeros((N,) + y0.shape)
    y[0,...] = y0
    for k in range(N-1):
        y1,y2 = y[k]
        # NOTE: Sympy is generating bad code, that doesnt want to let it reshape itself
        # y[k+1] = expo_euler_step(y[k], h, f, np.array([[2,2],[0,-1]]) )
        # y[k+1] = expo_euler_step(y[k], h, f, Df(y1,y2).reshape(2,2))
        y[k+1] = expo_euler_step(y[k], h, f, Df(y1,y2))
    return y, t


if __name__ == "__main__":
    y1,y2 = sp.symbols("y1,y2")

    ### Da hattest du einen Fehler in der RHS (-y2 am Schluss statt -y1)
    f_sp = sp.Matrix([-y1**2/y2+ y2* sp.log(y2), -y1])
    Df_sp = f_sp.jacobian((y1,y2))
    
    f0 = sp.lambdify((y1, y2), f_sp)
    f = lambda *x : np.reshape(f0(*x), (2,1))
    Df0 = sp.lambdify((y1, y2), Df_sp)
    Df = lambda *x : np.reshape(Df0(*x), (2,2)).astype(np.double)

    y0 = np.array([-1.,1.]).reshape(2,1)

    y,t = wrapper(f, Df, 48, y0, 6)

    #plt.plot(*y)
    plt.plot(t,y[:,0],label=r'$y_1$')
    plt.plot(t,y[:,1],label=r'$y_2$')
    plt.show()