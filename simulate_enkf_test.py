import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from src.env.sim import Simulation
from src.env.util import generate_comparison_gif, generate_contourf_gif
from src.EnKF.util import generate_observation_matrix, compute_l2_norm
from src.EnKF.enkf import EnsembleKalmanFilter

def parsing():
    parser = argparse.ArgumentParser(description="Data assimilation for estimating the dynamics of MHD turbulence")

    # Simulator setup
    parser.add_argument("--num_mesh", type=int, default=50)
    parser.add_argument("--t_end", type=float, default=0.8)
    parser.add_argument("--L", type=float, default=5.0)
    parser.add_argument("--courant_factor", type=float, default=0.25)
    parser.add_argument("--slopelimit", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--plot_freq", type=int, default=10)
    parser.add_argument("--plot_all", type=bool, default=False)
    parser.add_argument("--savedir", type=str, default="./results/test")

    # Kalman-Filter setup
    parser.add_argument("--num_data_point", type=int, default=100)
    parser.add_argument("--num_ensemble", type=int, default=100)
    parser.add_argument("--sigma_x", type=float, default=0.0001)
    parser.add_argument("--sigma_z", type=float, default=0.0001)

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    
    if not os.path.exists(args['savedir']):
        os.mkdir(args['savedir'])
        
    sim_gt = Simulation(
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
        plot_all=args["plot_all"],
    )

    sim_kf = Simulation(
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
        plot_all=args["plot_all"],
        use_smooth=True
    )

    # Generate simulation data : ground-truth
    sim_gt.dt_min = 0.001
    sim_gt.solve()

    rho_gt = sim_gt.record
    ts_measure = np.linspace(0, args['t_end'], len(rho_gt) + 1, endpoint = True)[:-1]

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
        return sim_kf.update(rho = rho, update_params = False)[0]

    x = sim_kf.rho.reshape(-1,1)

    enkf = EnsembleKalmanFilter(x, fx, N, xdim, zdim, H, Q, R, P)

    # simulatio with kalman filtering
    t = 0
    t_end = args['t_end']
    count = 0

    rho_measure = [rho_gt[0]]
    rho_estimate = [sim_kf.rho]
    t_estimate = [0]

    print("======================================================================")
    print("# Ensemble Kalman Filter Application for Constraint Transport Solver")
    while t < t_end:
            
        # Save prior state
        x_prev = enkf.x.reshape(args["num_mesh"], args["num_mesh"])

        # Kalman filter to estimate x
        x = enkf.predict()
        y_pred = H @ x

        # Update simulators for new parameters
        sim_kf.update(rho = x_prev, update_params=True)

        # Find index for measure
        idx = np.argmin(np.abs(ts_measure - t))
        x_gt = rho_gt[idx]
        
        y_true = H @ x_gt.reshape(-1,1)
        enkf.update(y_true)

        # update time
        t+= sim_kf.dt
        count += 1
        
        # if count > 30:
        #      breakpoint()

        # save trajectory
        t_estimate.append(t)
        rho_measure.append(x_gt)
        rho_estimate.append(sim_kf.rho)

        if t >= t_end:
            break

        if count % args['verbose'] == 0:
            print("t = {:.3f} | divB = {:.3f} | E = {:.3f} | P = {:.3f} | rho = {:.3f}".format(t, np.mean(np.abs(sim_kf.divB)), np.mean(sim_kf.en), np.mean(sim_kf.P), np.mean(sim_kf.rho)))

    generate_comparison_gif(rho_measure, rho_estimate, args['savedir'], "density_evolution_comparison.gif", None, args['plot_freq'])
    generate_contourf_gif(rho_measure,args['savedir'], "density_evolution_original.gif", r"$\rho (x,y)$", 0, args['L'], args['plot_freq'])
    generate_contourf_gif(rho_estimate,args['savedir'], "density_evolution_estimate.gif", r"$\rho (x,y)$", 0, args['L'], args['plot_freq'])

    l2_err_t = [compute_l2_norm(rho_measure[i], rho_estimate[i]) for i in range(len(rho_measure))]

    # Plot the L2 norm
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot(t_estimate, l2_err_t, "r")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L2 error")
    ax.set_title(r"L2 error with $\rho$ estimation")

    fig.tight_layout()
    fig.savefig(os.path.join(args['savedir'],"enkf_density_error.png"))