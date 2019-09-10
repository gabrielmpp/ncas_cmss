import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt


class Courrant():
    '''
    Class to compute the courrant number for the
    '''

    def __init__(self, mode, dt, dx, constant_u):
        self.mode = mode
        self.dt = dt
        self.dx = dx
        self.linear_courrant = constant_u * dx / dx

    def linear(self, *args):
        return self.linear_courrant

    def nonlinear(self, dt, dx, u):
        return u * dt / dx


def initialCondition(x):
    return np.where(x % 1 < 0.5, np.power(np.sin(2 * x * np.pi), 2), 0)


def Diffusion(phi_downwind, phi_upwind, phi, dx, dt, Diff=True):
    """
    Function to compute the diffusion

    :param phi_downwind: np.array
    :param phi_upwind: np.array
    :param phi: np.array
    :param dx: float
    :param Diff: Boolean
    :return: np.array
    """
    diff_constant = 1e-2

    if not Diff:
        diffusion = 0
    else:
        diffusion_courrant = diff_constant * dt / dx ** 2
        print(f"Diffusion stability factor is {diffusion_courrant}")
        if diffusion_courrant > 0.25:
            warn("Diffusion not stable")
        diffusion = diffusion_courrant * (phi_downwind - 2 * phi + phi_upwind)
    return diffusion


def main(mode='linear', Diff=False):
    # ---- Fixed parameters ----#
    nx = 20
    x = np.linspace(0, 1, nx + 1)
    fixed_u = 1  # for linear advection
    nt = 2000
    dx = 1 / nx
    dt = 0.5e-3

    # ---- Initializing appropriate method for computing c*du/dx ---- #
    courrant = Courrant('linear', dt, dx, fixed_u)
    if mode == 'linear':
        c = courrant.linear
    elif mode == 'nonlinear':
        c = Courrant.nonlinear
    else:
        raise ValueError(f"Mode {mode} not supported")

    # ---- Initializing xr.DataArray to store model outputs ---- #
    array = xr.DataArray(np.zeros([int(nt), int(nx + 1)]), dims=['time', 'x'],
                         coords={'time': np.arange(nt) * dt, 'x': x})

    phi = initialCondition(x)
    phiNew = phi.copy()
    phiOld = phi.copy()

    # FTCS for the first time-step
    for j in range(1, nx):
        phi[j] = phiOld[j] - 0.5 * c(dt, dx, phi[j]) * (phiOld[j + 1] - phiOld[j - 1]) + \
                 Diffusion(phi[j + 1], phi[j - 1], phi[j], dx, dt, Diff)

    phi[0] = phiOld[0] - 0.5 * c(dt, dx, phi[0]) * (phiOld[1] - phiOld[nx - 1]) + \
             Diffusion(phi[1], phi[nx - 1], phi[0], dx, dt, Diff)
    phi[nx] = phi[0]
    array[0, :] = phi

    for n in range(1, nt):
        for j in range(1, nx):
            phiNew[j] = phiOld[j] - c(dt, dx, phi[j]) * (phi[j + 1] - phi[j - 1]) + \
                        Diffusion(phi[j + 1], phi[j - 1], phi[j], dx, dt, Diff)
        phiNew[0] = phiOld[0] - c(dt, dx, phi[0]) * (phi[1] - phi[nx - 1]) + \
                    Diffusion(phi[1], phi[nx - 1], phi[0], dx, dt, Diff)
        phiNew[nx] = phiNew[0]
        array[n, :] = phi
        phiOld = phi.copy()
        phi = phiNew.copy()
    print(array)
    array.plot(x='x', cmap='nipy_spectral', vmin=0)
    plt.show()


if __name__ == "__main__":
    main(mode='nonlinear', Diff=False)
