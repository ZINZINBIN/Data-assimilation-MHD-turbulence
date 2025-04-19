import argparse, os
import numpy as np
from src.env.sim import Simulation
from src.env.util import generate_comparison_gif
from src.EnKF.util import generate_observation_matrix
from src.EnKF.enkf import EnsembleKalmanFilter

def parsing():
    parser = argparse.ArgumentParser(description="Data assimilation for estimating the dynamics of MHD turbulence")

    # Simulator setup
    parser.add_argument("--num_mesh", type=int, default=50)
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--courant_factor", type=float, default=0.1)
    parser.add_argument("--slopelimit", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--plot_freq", type=int, default=20)
    parser.add_argument("--plot_all", type=bool, default=False)
    parser.add_argument("--savedir", type=str, default="./results/")

    # Kalman-Filter setup
    parser.add_argument("--num_data_point", type=int, default=100)
    parser.add_argument("--num_ensemble", type=int, default=100)
    parser.add_argument("--sigma_x", type=float, default=0.005)
    parser.add_argument("--sigma_z", type=float, default=0.005)

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    sim = Simulation(
        nx=args["num_mesh"],
        ny=args["num_mesh"],
        t_end=args["t_end"],
        L=args["L"],
        slopelimit=args["slopelimit"],
        animation=args["use_animation"],
        verbose=args["verbose"],
        savedir=args["savedir"],
        plot_freq=args["plot_freq"],
        courant_factor=args["courant_factor"],
        plot_all=args["plot_all"]
    )
    
    sim.solve()
    rho_gt = sim.record
    
    # initialize
    sim.set_init_condition()

    # Kalman Filter
    N = args['num_ensemble']
    sig_x = args['sigma_x']
    sig_z = args['sigma_z']
    xdim = args['num_mesh'] ** 2
    zdim = args['num_data_point']

    H = generate_observation_matrix(args['num_mesh'], args['num_data_point'])
    Q = np.eye(xdim) * sig_x ** 2
    R = np.eye(zdim) * sig_z ** 2
    P = np.eye(xdim) * sig_x ** 2

    def fx(x:np.ndarray):
        rho = x.reshape(args['num_mesh'], args['num_mesh'])
        return sim.update(rho = rho)[0].ravel()

    x = sim.rho.reshape(-1,1)

    enkf = EnsembleKalmanFilter(x, fx, N, xdim, zdim, H, Q, R, P)

    # simulatio with kalman filtering
    t = 0
    t_end = args['t_end']
    count = 0

    rho_estimate = [sim.rho]

    while t < t_end:
        x = enkf.predict()
        y_pred = H @ x

        y_true = H @ sim.rho.reshape(-1,1)
        enkf.update(y_true)

        rho_estimate.append(sim.rho)

        t+= sim.dt
        count += 1

        if t >= t_end:
            break

        if count % args['verbose'] == 0:
            print("t = {:.3f} | divB = {:.3f} | E = {:.3f} | P = {:.3f} | rho = {:.3f}".format(t, np.mean(np.abs(sim.divB)), np.mean(sim.en), np.mean(sim.P), np.mean(sim.rho)))

    generate_comparison_gif(rho_gt, rho_estimate, args['savedir'], "density_evolution_comparison.gif", None, args['plot_freq'])
