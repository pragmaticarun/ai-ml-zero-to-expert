"""Day 15: derivatives as slopes; numerical differentiation and its error."""
import numpy as np
f = lambda x: x**3 - 2*x; df = lambda x: 3*x**2 - 2
x0 = 1.5; print("analytic f'(1.5) =", df(x0))
for h in [1e-1, 1e-3, 1e-5, 1e-8, 1e-12]:
    fwd = (f(x0+h) - f(x0)) / h; cen = (f(x0+h) - f(x0-h)) / (2*h)
    print(f"h={h:.0e}: forward err={abs(fwd-df(x0)):.2e}  central err={abs(cen-df(x0)):.2e}")
# derivative of the squared-error cost for 1 data point: J(w) = (w*x - y)^2
x, y = 3.0, 12.0; J = lambda w: (w*x - y)**2; dJ = lambda w: 2*(w*x - y)*x
for w in [2.0, 4.0, 5.0]: print(f"w={w}: J={J(w):.1f}, dJ/dw={dJ(w):+.1f} -> move w {'up' if dJ(w)<0 else 'down' if dJ(w)>0 else 'nowhere'}")
