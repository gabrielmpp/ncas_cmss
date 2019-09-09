import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt


def initialCondition(x):
    return np.where(x % 1 < 0.5, np.power(np.sin(2 * x * np.pi), 2), 0)


def courrant(dt, dx, mode='linear', u=None):
    if mode == 'linear':
        c = 1 * dt / dx
    elif mode == 'nonlinear':
        c = u * dt / dx
    print(c)
    if c > 0.25:
        warn("Courrant number larger than 0.25!")
    print(f"Courrant number is {c}")

    return c


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
        print(f"Diffusiant currant is {diffusion_courrant}")
        if diffusion_courrant > 0.25: warn("Diffusion not stable")
        diffusion = diffusion_courrant * (phi_downwind - 2 * phi + phi_upwind)
    return diffusion


def main(mode='linear', Diff=False):
    nx = 20
    c = 0.2
    x = np.linspace(0, 1, nx + 1)
    u = 1
    nt = 2000
    dx = 1 / nx
    dt = 0.5e-3
    array = xr.DataArray(np.zeros([int(nt), int(nx + 1)]), dims=['time', 'x'],
                         coords={'time': np.arange(nt) * dt, 'x': x})

    phi = initialCondition(x)
    phiNew = phi.copy()
    phiOld = phi.copy()

    # FTCS for the first time-step
    for j in range(1, nx):
        phi[j] = phiOld[j] - 0.5 * courrant(dt, dx, u=phiOld[j], mode=mode) * (phiOld[j + 1] - phiOld[j - 1]) + \
                 Diffusion(phi[j + 1], phi[j - 1], phi[j], dx, dt, Diff)

    phi[0] = phiOld[0] - 0.5 * courrant(dt, dx, u=phiOld[j], mode=mode) * (phiOld[1] - phiOld[nx - 1]) + \
             Diffusion(phi[1], phi[nx - 1], phi[0], dx, dt, Diff)
    phi[nx] = phi[0]
    array[0, :] = phi

    for n in range(1, nt):
        for j in range(1, nx):
            phiNew[j] = phiOld[j] - courrant(dt, dx, u=phi[j], mode=mode) * (phi[j + 1] - phi[j - 1]) + \
                        Diffusion(phi[j + 1], phi[j - 1], phi[j], dx, dt, Diff)
        phiNew[0] = phiOld[0] - courrant(dt, dx, u=phi[j], mode=mode) * (phi[1] - phi[nx - 1]) + \
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
