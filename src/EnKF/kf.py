import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(
        self, 
        F:np.ndarray, 
        H:Optional[np.ndarray] = None, 
        Q:Optional[np.ndarray] = None, 
        R:Optional[np.ndarray] = None, 
        P:Optional[np.ndarray] = None,
        B:Optional[np.ndarray] = None,
        x0:Optional[np.ndarray] = None
        ):

        self.F = F
        self.H = H

        # dims
        self.n = F.shape[1]
        self.m = H.shape[0]

        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R     
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.B = 0 if B is None else B
        
    def predict(self, u:np.ndarray):
        xf = self.F @ self.x + self.B @ u
        Pf = self.F @ self.P @ self.F.T + self.Q
        self.x = xf
        self.P = Pf
        return xf
    
    def update(self, y:np.ndarray):
        res = y - self.H @ self.x
        S = self.R + self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        x = self.x + K@res
        P = (np.eye(self.n) - K@self.H)@self.P
        
        self.x = x
        self.P = P


if __name__ == "__main__":

    # setup
    dt = 0.01
    sig_a = 0.25
    sig_z = 0.25
    w = 1.0

    # Matrix for describing dynamics
    F = np.array([[1,dt],[0,1]])
    G = np.array([0.5 * dt ** 2, dt]).reshape(-1,1)
    Q = G@G.T * sig_a ** 2
    H = np.array([1,0]).reshape(1,2)
    P = np.array([[sig_a ** 2, 0], [0, sig_a ** 2]])
    R = np.eye(1) * sig_z ** 2
    B = np.array([0.5 * dt ** 2, dt]).reshape(-1,1)

    x0 = np.array([0,4]).reshape(-1,1)
    y_true = 0

    # Kalman Filter
    kf = KalmanFilter(F, H, Q, R, P, B, x0)

    traj_estimate = []

    # Applying the Kalman Filter
    N_iter = 1000

    def f(u):
        a = np.array([-(w**2) * u]).reshape(1,1)
        return a

    def m(t):
        return 4 * np.sin(w*t) + np.random.normal(0, sig_z)

    measurements = [m(t) for t in [i * dt for i in range(N_iter)]]

    for i in np.arange(0, N_iter):
        x = kf.predict(f(y_true))
        y_pred = H@x

        traj_estimate.append(y_pred)

        y_true = measurements[i]
        kf.update(y_true)

    measurements = np.array(measurements)
    predictions = np.concatenate(traj_estimate, axis = 1).T

    plt.plot(measurements, label = 'Measurements')
    plt.plot(predictions[:,0], label = 'Kalman Filter Prediction')
    plt.legend()
    plt.savefig("./results/test/test_kf.png")