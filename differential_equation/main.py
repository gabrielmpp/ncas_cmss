import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt


class Courrant():
    '''
    Class with methods to compute the Courrant number for linear and nonlinear advection
    '''

    def __init__(self, mode, dt, dx, constant_u):
        """
        :param mode: str
        :param dt: float
        :param dx: float
        :param constant_u: float
        """
        self.mode = mode
        self.dt = dt
        self.dx = dx
        self.linear_courrant = constant_u * dt / dx

    def linear(self, *args, **kwargs):
        return self.linear_courrant

    def nonlinear(self, u):
        return u * self.dt / self.dx


def initialCondition_2(x):
    return np.where(x % 1 < 0.5, np.power(np.sin(2 * x * np.pi), 2), 1e-5)

def initialCondition(x):
    return np.power(np.sin(2 * x * np.pi), 2)

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
    nx = 50
    output_rate = 10 # output saved every N timesteps
    x = np.linspace(0, 1, nx + 1)
    fixed_u = 1  # for linear advection
    nt = 500
    dx = 1 / nx
    dt = 0.5e-3

    # ---- Initializing appropriate method for computing c*du/dx ---- #
    courrant = Courrant('linear', dt, dx, fixed_u)
    if mode == 'linear':
        c = courrant.linear
    elif mode == 'nonlinear':
        c = courrant.nonlinear
    else:
        raise ValueError(f"Mode {mode} not supported")

    # ---- Initializing xr.DataArray to store model outputs ---- #
    time_coord = np.arange(0, nt, output_rate) * dt
    array = xr.DataArray(np.zeros([time_coord.shape[0], int(nx + 1)]), dims=['time', 'x'],
                         coords={'time': time_coord, 'x': x})

    phi = initialCondition(x)
    phiNew = phi.copy()
    phiOld = phi.copy()

    # ---- FTCS for the first time-step ---- #
    for j in range(1, nx):
        phi[j] = phiOld[j] - 0.5 * c(phi[j]) * (phiOld[j + 1] - phiOld[j - 1]) + \
                 Diffusion(phi[j + 1], phi[j - 1], phi[j], dx, dt, Diff)

    phi[0] = phiOld[0] - 0.5 * c(u=phi[0]) * (phiOld[1] - phiOld[nx - 1]) + \
             Diffusion(phiOld[1], phiOld[nx - 1], phiOld[0], dx, dt, Diff)
    phi[nx] = phi[0]

    # ---- CTCS integration ---- #
    for n in range(1, nt):
        for j in range(1, nx):
            phiNew[j] = phiOld[j] - c(phi[j]) * (phi[j + 1] - phi[j - 1]) + \
                        Diffusion(phiOld[j + 1], phiOld[j - 1], phiOld[j], dx, 2*dt, Diff)

        phiNew[0] = phiOld[0] - c(phi[0]) * (phi[1] - phi[nx - 1]) + \
                    Diffusion(phi[1], phi[nx - 1], phi[0], dx, dt, Diff)
        phiNew[nx] = phiNew[0]
        if n % output_rate == 0:
            array.loc[n*dt, ] = phi

        phiOld = phi.copy()
        phi = phiNew.copy()

    # ---- Plotting ---- #

    array.plot(x='x', cmap='nipy_spectral', vmin=0)
    plt.show()
    plt.close()
    array.isel(time=1).plot()
    array.plot.line(x='x')
    plt.show()


if __name__ == "__main__":
    main(mode='nonlinear', Diff=True)
