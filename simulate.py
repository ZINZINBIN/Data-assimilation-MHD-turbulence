import argparse, os
import numpy as np
from src.env.sim import Simulation

def parsing():
    parser = argparse.ArgumentParser(description="Data assimilation for estimating the dynamics of MHD turbulence")

    # Simulator setup
    parser.add_argument("--num_mesh", type=int, default=128)
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--courant_factor", type=float, default=0.4)
    parser.add_argument("--slopelimit", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=True)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--plot_freq", type=int, default=40)
    parser.add_argument("--plot_all", type=bool, default=True)
    parser.add_argument("--savedir", type=str, default="./results/simulation")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    
    if not os.path.exists(args['savedir']):
        os.mkdir(args['savedir'])
        
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
        plot_all=args['plot_all']
    )
    sim.solve()

    # save the results for analysis
    np.save("./data/Ez.npy", sim.Ez)
    np.save("./data/J.npy", sim.J)
    np.save("./data/w.npy", sim.w)
    np.save("./data/Pm.npy", sim.Pm)
    np.save("./data/px.npy", sim.px)
    np.save("./data/py.npy", sim.py)
    np.save("./data/en.npy", sim.en)
    np.save("./data/vx.npy", sim.vx)
    np.save("./data/vy.npy", sim.vy)
    np.save("./data/rho.npy", sim.rho)
    np.save("./data/P.npy", sim.P)
    np.save("./data/Bx.npy", sim.Bx)
    np.save("./data/By.npy", sim.By)