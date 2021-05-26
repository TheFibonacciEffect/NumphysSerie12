# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.linalg import expm
# %%
def expo_euler_step(yk, h, f, J):
    z = h*J
    A = expm(z) - np.eye(2)
    b = z @ f(*yk)
    prop = np.linalg.solve(A, b)
    return yk + h*prop

def wrapper(f, Df, N, y0, t):
    t, h = np.linspace(0,t, N, retstep=True)
    y = np.zeros((N,) + y0.shape)
    y[0,...] = y0
    for k in range(N-1):
        y[k+1] = expo_euler_step(y[k], h, f, Df(*y[k]))
    return y, t

solution = lambda t: np.array([-np.cos(t) * np.exp(np.sin(t)), np.exp(np.sin(t))])
if __name__ == "__main__":
    y1,y2 = sp.symbols("y1,y2")

    ### Da hattest du einen Fehler in der RHS (-y2 am Schluss statt -y1)
    f_sp = sp.Matrix([-y1**2/y2+ y2* sp.log(y2), -y1])
    Df_sp = f_sp.jacobian((y1,y2))
    
    f0 = sp.lambdify((y1, y2), f_sp)
    f = lambda *x : np.reshape(f0(*x), (2,1)).astype(np.double)
    Df0 = sp.lambdify((y1, y2), Df_sp)
    Df = lambda *x : np.reshape(Df0(*x), (2,2)).astype(np.double)

    y0 = np.array([-1.,1.]).reshape(2,1)

    T = 6
    y,t = wrapper(f, Df, 48, y0, T)
    plt.plot(t, y[:, 0])
    plt.clf()

    n = 5
    error = np.zeros(n)
    timesteps = np.geomspace(24, 384, 5).astype(int)
    for i,k in enumerate(timesteps):
        y,t = wrapper(f, Df, k, y0, T)
        error[i] = np.linalg.norm(y[-1,:] - solution(t))
    print(error)
    plt.loglog(timesteps, error)
    plt.show()