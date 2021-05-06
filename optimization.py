import numpy as np
import matplotlib.pyplot as plt


class CurveFittingExample:
    def __init__(self):
        self.y = np.linspace(-5, 5, 100)
        self.n_meas_points = 100
        self.n_params = 3

        self.x_ref = [0.7, 0.1, 0.7] # np.random.random(self.n_params)
        self.noise = 0.0
        self.m_meas = self.model(self.y, self.x_ref) + np.random.randn(self.n_meas_points)*self.noise

    def model(self, y, x):
        return x[0]**2. * np.sin(y) + x[1]**3. * np.exp(y) + x[2] ** 3. * y * x[1]

    def jac_model(self, y, x):
        jac = np.zeros((y.shape[0], x.shape[0]))
        jac[:, 0] = np.sin(y)*2.*x[0]
        jac[:, 1] = np.exp(y)*3.*x[1]**2. + x[2] ** 3. * y
        jac[:, 2] = 3.*x[2]**2.*y*x[1]
        return jac

    def f(self, x):
        return self.model(self.y, x) - self.m_meas

    def F(self, x):
        return 1. / 2. * np.sum(np.square(self.f(x)))

    def J(self, x):
        return self.jac_model(self.y, x)

    def n_parameters(self):
        return self.n_params


# ALGORITHM
def gauss_newton(F, f, J, x0, k_max, callback=None):
    def search_direction(x):
        jac = J(x)
        fct = f(x)

        A = np.matmul(jac.T, jac)
        b = np.matmul(-jac.T, fct)
        hgn = np.linalg.solve(A, b)
        return hgn, True

    def step_length(p, hd):
        return 1.

    k = 0
    x = x0
    found = False
    while not found and k < k_max:
        h_d, exists = search_direction(x)
        if not exists:
            found = True
        else:
            alpha = step_length(x, h_d)
            x = x + alpha*h_d
            if callback:
                callback(x)
            k = k + 1
    return x, k


def levenberg_marquard(F, f, J, x0, k_max, eps_1=1.e-15, eps_2=1.e-15, tau=1., f_and_J=None, callback=None):
    def inf_norm(x):
        return np.linalg.norm(x, np.inf)

    def norm(x):
        return np.linalg.norm(x, 2)
    k = 0
    v = 2.
    x = x0
    F_x = F(x)
    if f_and_J is not None:
        fct, jac = f_and_J(x)
    else:
        jac = J(x)
        fct = f(x)
    if callback:
        callback(x, fct, 0)
    A = np.matmul(jac.T, jac)
    g = np.matmul(jac.T, fct)

    # def L(h):
    #     return F_x + np.matmul(np.matmul(h.T, jac.T), fct) + 1./2.*np.matmul(np.matmul(np.matmul(h.T, jac.T), jac), h)
    found = inf_norm(g) <= eps_1
    mu = tau*np.max(A)
    while not found and k < k_max:
        k = k+1
        h_lm = np.linalg.solve(A + mu*np.eye(A.shape[0]), -g)
        if norm(h_lm) <= eps_2*(norm(x) + eps_2):
            found = True
        else:
            x_new = x + h_lm
            F_x_new = F(x_new)
            # rho = (F_x - F_x_new)/(L(np.zeros(h_lm.shape)) - L(h_lm))
            rho = (F_x - F_x_new)/(0.5*np.dot(h_lm, mu*h_lm - g))
            if rho > 0:
                x = x_new
                F_x = F_x_new
                if f_and_J is not None:
                    fct, jac = f_and_J(x)
                else:
                    jac = J(x)
                    fct = f(x)
                if callback:
                    callback(x, fct, k)
                A = np.matmul(jac.T, jac)
                g = np.matmul(jac.T, fct)
                found = inf_norm(x) <= eps_1
                mu = mu*max(1./3., 1. - (2.*rho - 1.)**3.)
                v = 2.
            else:
                mu = mu*v
                v = 2.*v
    return x, k


def example():
    cfe = CurveFittingExample()
    x0 = np.random.random(cfe.n_parameters())

    n_iter = 100

    gn_history = []
    lm_history = []
    x_gn, k_gn = gauss_newton(cfe.F, cfe.f, cfe.J, x0, n_iter, lambda x: gn_history.append(x))


    def lm_callback(x, _):
        lm_history.append(x)


    x_lm, k_lm = levenberg_marquard(cfe.F, cfe.f, cfe.J, x0, n_iter, 1e-15, 1e-15, 1., None, lm_callback)

    print("reference parameter: {}".format(cfe.x_ref))
    print("gauss newton minimum: {} steps: {}".format(x_gn, k_gn))
    print("levenberg marquard minimum: {} steps: {}".format(x_lm, k_lm))

    plt.figure()
    for p in gn_history[:-1]:
        plt.plot(p[0], p[1], 'o', color="red")
    if gn_history:
        plt.plot(gn_history[-1][0], gn_history[-1][1], 'o', color="red", label="gn")
    for p in lm_history[:-1]:
        plt.plot(p[0], p[1], 'o', color="blue")
    plt.plot(lm_history[-1][0], lm_history[-1][1], 'o', color="blue", label="lm")
    plt.plot(cfe.x_ref[0], cfe.x_ref[1], 'o', color="black", label="ref")
    plt.plot(x0[0], x0[1], 'o', color="green", label="x0")

    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    ff = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            ff[i, j] = cfe.F(np.array([xx[i, j], yy[i, j], cfe.x_ref[2]]))
    plt.contourf(xx, yy, ff, 100)
    plt.axis("equal")
    plt.legend()
    plt.show()

    y_ = np.linspace(-5, 5, 500)
    plt.figure()
    plt.plot(y_, cfe.model(y_, cfe.x_ref), label="reality")
    plt.plot(cfe.y, cfe.m_meas, 'o', label="measure")
    plt.plot(y_, cfe.model(y_, x_gn), label="reconstruction_gauss_newton")
    plt.plot(y_, cfe.model(y_, x_lm), label="reconstruction_levenberg_marquard")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    example()