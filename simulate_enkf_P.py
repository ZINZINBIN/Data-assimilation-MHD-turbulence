import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from src.env.sim import Simulation
from src.env.util import generate_comparison_gif, generate_contourf_gif, generate_snapshots, generate_comparison_snapshot
from src.EnKF.util import generate_observation_matrix, compute_l2_norm
from src.EnKF.enkf import EnsembleKalmanFilter

def parsing():
    parser = argparse.ArgumentParser(description="Data assimilation for estimating the dynamics of MHD turbulence")

    # Simulator setup
    parser.add_argument("--num_mesh", type=int, default=50)
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--courant_factor", type=float, default=0.25)
    parser.add_argument("--slopelimit", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--plot_freq", type=int, default=10)
    parser.add_argument("--plot_all", type=bool, default=False)
    parser.add_argument("--savedir", type=str, default="./results/P")

    # Kalman-Filter setup
    parser.add_argument("--num_data_point", type=int, default=100)
    parser.add_argument("--num_ensemble", type=int, default=100)
    parser.add_argument("--sigma_x", type=float, default=0.01)
    parser.add_argument("--sigma_z", type=float, default=0.01)

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()

    savepath = os.path.join(
        args["savedir"],
        "np_{}_ne_{}_sx_{}_sz_{}".format(
            args["num_data_point"],
            args["num_ensemble"],
            args["sigma_x"],
            args["sigma_z"],
        ),
    )

    if not os.path.exists(savepath):
        os.makedirs(savepath)

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

    P_gt = sim_gt.record_P
    ts_measure = np.linspace(0, args['t_end'], len(P_gt) + 1, endpoint = True)[:-1]

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
        p = x.reshape(args['num_mesh'], args['num_mesh'])
        return sim_kf.update(P = p, update_params = False)[0]

    x = sim_kf.P.reshape(-1,1)

    enkf = EnsembleKalmanFilter(x, fx, N, xdim, zdim, H, Q, R, P)

    # simulatio with kalman filtering
    t = 0
    t_end = args['t_end']
    count = 0

    P_measure = [P_gt[0]]
    P_estimate = [sim_kf.P]
    t_estimate = [0]

    record_rho = []
    record_vx = []
    record_vy = []
    record_P = []
    record_Bx = []
    record_By = []

    print("======================================================================")
    print("# Ensemble Kalman Filter Application for Constraint Transport Solver")
    while t < t_end:

        # record physical variables
        record_rho.append(np.copy(sim_kf.rho))
        record_vx.append(np.copy(sim_kf.vx))
        record_vy.append(np.copy(sim_kf.vy))
        record_P.append(np.copy(sim_kf.P))
        record_Bx.append(np.copy(sim_kf.Bx))
        record_By.append(np.copy(sim_kf.By))

        # Save prior state
        x_prev = enkf.x.reshape(args["num_mesh"], args["num_mesh"])

        # Kalman filter to estimate x
        x = enkf.predict()
        y_pred = H @ x

        # Update simulators for new parameters
        sim_kf.update(P = x_prev, update_params=True)

        # Find index for measure
        idx = np.argmin(np.abs(ts_measure - t))
        x_gt = P_gt[idx]

        y_true = H @ x_gt.reshape(-1,1)
        enkf.update(y_true)

        # update time
        t+= sim_kf.dt
        count += 1

        # save trajectory
        t_estimate.append(t)
        P_measure.append(x_gt)
        P_estimate.append(sim_kf.P)

        if t >= t_end:
            break

        if count % args['verbose'] == 0:
            print("t = {:.3f} | divB = {:.3f} | E = {:.3f} | P = {:.3f} | rho = {:.3f}".format(t, np.mean(np.abs(sim_kf.divB)), np.mean(sim_kf.en), np.mean(sim_kf.P), np.mean(sim_kf.rho)))

    generate_comparison_gif(P_measure, P_estimate, savepath, "pressure_evolution_comparison.gif", None, args['plot_freq'])
    generate_contourf_gif(P_measure,  savepath, "pressure_evolution_original.gif", r"$P(x,y)$", 0, args['L'], args['plot_freq'])
    generate_contourf_gif(P_estimate, savepath, "pressure_evolution_estimate.gif", r"$P(x,y)$", 0, args['L'], args['plot_freq'])

    generate_snapshots(record_rho, r"$\rho(x,y)$", savepath, "rho_snapshot.png")
    generate_snapshots(record_vx, r"$v_x(x,y)$", savepath, "vx_snapshot.png")
    generate_snapshots(record_vy, r"$v_y(x,y)$", savepath, "vy_snapshot.png")
    generate_snapshots(record_P, r"$P(x,y)$", savepath, "pressure_snapshot.png")
    generate_snapshots(record_Bx, r"$B_x(x,y)$", savepath, "Bx_snapshot.png")
    generate_snapshots(record_By, r"$B_y(x,y)$", savepath, "By_snapshot.png")
    
    generate_comparison_snapshot(P_measure[-1], P_estimate[-1], savepath, "pressure_comparison.png", r"$P(x,y)$ at $t = 0.5$")

    l2_err_t = [compute_l2_norm(P_measure[i], P_estimate[i]) for i in range(len(P_measure))]

    # Plot the L2 norm
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot(t_estimate, l2_err_t, "r")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L2 error")
    ax.set_title(r"L2 error with $P$ estimation")

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "enkf_pressure_error.png"))

    try:
        datapath = os.path.join(
            "data/P",
            "np_{}_ne_{}_sx_{}_sz_{}".format(
                args["num_data_point"],
                args["num_ensemble"],
                args["sigma_x"],
                args["sigma_z"],
            ),
        )

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        t_estimate = np.array(t_estimate)
        P_estimate = np.stack(P_estimate, axis=2)
        P_measure = np.stack(P_measure, axis=2)

        np.save(os.path.join(datapath, "t_estimate.npy"), t_estimate)
        np.save(os.path.join(datapath, "pressure_estimate.npy"), P_estimate)
        np.save(os.path.join(datapath, "pressure_measure.npy"), P_measure)

    except:
        print("NaN or invalid value contained in EnKF with pressure simulation")
