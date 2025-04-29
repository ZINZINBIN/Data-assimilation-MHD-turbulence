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
    parser.add_argument("--use_smooth", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--plot_freq", type=int, default=10)
    parser.add_argument("--plot_all", type=bool, default=False)
    parser.add_argument("--savedir", type=str, default="./results/v")

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
        use_smooth=args["use_smooth"],
    )

    # Generate simulation data : ground-truth
    sim_gt.dt_min = 0.001
    sim_gt.solve()

    vx_gt = sim_gt.record_vx
    vy_gt = sim_gt.record_vy

    ts_measure = np.linspace(0, args['t_end'], len(vx_gt) + 1, endpoint = True)[:-1]

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

    def fx_vx(x:np.ndarray):
        vx = x.reshape(args['num_mesh'], args['num_mesh'])
        return sim_kf.update(vx = vx, update_params = False)[4]

    def fx_vy(x:np.ndarray):
        vy = x.reshape(args['num_mesh'], args['num_mesh'])
        return sim_kf.update(vy = vy, update_params = False)[4]

    vx = sim_kf.vx.reshape(-1,1)
    vy = sim_kf.vy.reshape(-1,1)

    enkf_vx = EnsembleKalmanFilter(vx, fx_vx, N, xdim, zdim, H, Q, R, P)
    enkf_vy = EnsembleKalmanFilter(vy, fx_vy, N, xdim, zdim, H, Q, R, P)

    # simulatio with kalman filtering
    t = 0
    t_end = args['t_end']
    count = 0

    vx_measure = [vx_gt[0]]
    vy_measure = [vy_gt[0]]

    vx_estimate = [sim_kf.vx]
    vy_estimate = [sim_kf.vy]
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
        vx_prev = enkf_vx.x.reshape(args["num_mesh"], args["num_mesh"])
        vy_prev = enkf_vy.x.reshape(args['num_mesh'], args['num_mesh'])

        # Kalman filter to estimate x
        vx = enkf_vx.predict()
        vy = enkf_vy.predict()

        vx_obs_pred = H @ vx
        vy_obs_pred = H @ vy

        # Update simulators for new parameters
        sim_kf.update(vx = vx_prev, vy = vy_prev, update_params=True)

        # Find index for measure
        idx = np.argmin(np.abs(ts_measure - t))
        vx_true = vx_gt[idx]
        vy_true = vy_gt[idx]

        vx_obs_true = H @ vx_true.reshape(-1,1)
        vy_obs_true = H @ vy_true.reshape(-1,1)

        enkf_vx.update(vx_obs_true)
        enkf_vy.update(vy_obs_true)

        # update time
        t+= sim_kf.dt
        count += 1

        # save trajectory
        t_estimate.append(t)

        vx_measure.append(vx_true)
        vy_measure.append(vy_true)

        vx_estimate.append(sim_kf.vx)
        vy_estimate.append(sim_kf.vy)

        if t >= t_end:
            break

        if count % args['verbose'] == 0:
            print("t = {:.3f} | divB = {:.3f} | E = {:.3f} | P = {:.3f} | rho = {:.3f}".format(t, np.mean(np.abs(sim_kf.divB)), np.mean(sim_kf.en), np.mean(sim_kf.P), np.mean(sim_kf.rho)))

    generate_comparison_gif(vx_measure, vx_estimate, savepath, "vx_evolution_comparison.gif", None, args['plot_freq'])
    generate_comparison_gif(vy_measure, vy_estimate, savepath, "vy_evolution_comparison.gif", None, args['plot_freq'])

    generate_contourf_gif(vx_measure, savepath, "vx_evolution_original.gif", r"$v_x (x,y)$", 0, args['L'], args['plot_freq'])
    generate_contourf_gif(vy_measure, savepath, "vy_evolution_original.gif", r"$v_y (x,y)$", 0, args['L'], args['plot_freq'])

    generate_contourf_gif(vx_estimate, savepath, "vx_evolution_estimate.gif", r"$v_x (x,y)$", 0, args['L'], args['plot_freq'])
    generate_contourf_gif(vy_estimate, savepath, "vy_evolution_estimate.gif", r"$v_y (x,y)$", 0, args['L'], args['plot_freq'])

    generate_snapshots(record_rho, r"$\rho(x,y)$", savepath, "rho_snapshot.png")
    generate_snapshots(record_vx, r"$v_x(x,y)$", savepath, "vx_snapshot.png")
    generate_snapshots(record_vy, r"$v_y(x,y)$", savepath, "vy_snapshot.png")
    generate_snapshots(record_P, r"$P(x,y)$", savepath, "pressure_snapshot.png")
    generate_snapshots(record_Bx, r"$B_x(x,y)$", savepath, "Bx_snapshot.png")
    generate_snapshots(record_By, r"$B_y(x,y)$", savepath, "By_snapshot.png")

    generate_comparison_snapshot(vx_measure[-1], vx_estimate[-1], savepath, "vx_comparison.png", r"$v_x(x,y)$ at $t = 0.5$")
    generate_comparison_snapshot(vy_measure[-1], vy_estimate[-1], savepath, "vy_comparison.png", r"$v_y(x,y)$ at $t = 0.5$")

    vx_l2_err_t = [compute_l2_norm(vx_measure[i], vx_estimate[i]) for i in range(len(vx_measure))]
    vy_l2_err_t = [compute_l2_norm(vy_measure[i], vy_estimate[i]) for i in range(len(vy_measure))]

    # Plot the L2 norm
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot(t_estimate, vx_l2_err_t, "r", label = "$v_x$")
    ax.plot(t_estimate, vy_l2_err_t, "b", label = "$v_y$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L2 error")
    ax.set_title(r"L2 error with $v(x,y)$ estimation")

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "enkf_v_error.png"))

    try:
        datapath = os.path.join(
            "data/v",
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
        vx_estimate = np.stack(vx_estimate, axis=2)
        vy_estimate = np.stack(vy_estimate, axis=2)

        vx_measure = np.stack(vx_measure, axis=2)
        vy_measure = np.stack(vy_measure, axis=2)

        np.save(os.path.join(datapath, "t_estimate.npy"), t_estimate)
        np.save(os.path.join(datapath, "vx_estimate.npy"), vx_estimate)
        np.save(os.path.join(datapath, "vy_estimate.npy"), vy_estimate)

        np.save(os.path.join(datapath, "vx_measure.npy"), vx_measure)
        np.save(os.path.join(datapath, "vy_measure.npy"), vy_measure)

    except:
        print("NaN or invalid value contained in EnKF with velocity simulation")
