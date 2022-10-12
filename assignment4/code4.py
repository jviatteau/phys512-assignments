import numpy as np
import matplotlib.pyplot as plt

data = np.load('sidebands.npz')
t=data['time']
d=data['signal']

#Single Lorentzian fit

def calc_lorentz(m, t):
    a = m[0]
    w = m[1]
    t_0 = m[2]
    y = a*w**2 / (w**2+(t-t_0)**2)
    dyda = w**2/(w**2+(t-t_0)**2)
    dydw = 2*a*w*(t-t_0)**2/(w**2+(t-t_0)**2)**2
    dydt_0 = 2*a*w**2*(t-t_0)/(w**2+(t-t_0)**2)**2
    grad = np.zeros([t.size, m.size])
    grad[:, 0] = dyda
    grad[:, 1] = dydw
    grad[:, 2] = dydt_0
    return y, grad



def num_derivs(fun, t, params):
    derivs = np.zeros([t.size, params.size])
    for i in range(params.size):
        x = params[i]
        dx = np.abs((1e-16)**(1/3)*x)
        params_up = params.copy()
        params_up[i] = x+dx
        params_down = params.copy()
        params_down[i] = x-dx
        derivs[:,i] = (fun(t, params_up)-fun(t, params_down) )/(2*dx)
    return derivs

def lorentz(t, params):
    return params[0] / (1 + (t-params[2])**2/params[1]**2)

def calc_lorentz_num(fun, m, t):
    a = m[0]
    w = m[1]
    t_0 = m[2]
    y = fun(t, m)
    grad = num_derivs(fun, t, m)
    return y, grad

#params = [a, w, t_0, b, c, dt]
def lorentz_triple(t, params):
    A = params[0] / (1 + (t-params[2])**2/params[1]**2)
    B = params[3] / (1 + (t-params[2]+params[5])**2/params[1]**2)
    C = params[4] / (1 + (t-params[2]-params[5])**2/params[1]**2)
    return A+B+C


m = np.asarray([1.4, 0.00001, 0.0002]) #Initial guess

for j in range(10):
    pred, grad = calc_lorentz_num(lorentz, m, t)
    r = d - pred
    lhs = grad.T@grad
    rhs = grad.T@r
    dm = np.linalg.inv(lhs)@rhs
    m += dm

print(m)

m_triple = np.append(m, [0.2, 0.2, 5e-5])
pred3, grad3 = calc_lorentz_num(lorentz_triple, m_triple, t)

for j in range(10):
    pred3, grad3 = calc_lorentz_num(lorentz_triple, m_triple, t)
    r = d - pred3
    lhs = grad3.T@grad3
    rhs = grad3.T@r
    dm = np.linalg.inv(lhs)@rhs
    m_triple += dm

print(m_triple)

plt.plot(t*1000, d)
plt.plot(t*1000, pred)
plt.plot(t*1000, pred3)
plt.show()



