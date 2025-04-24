import numpy as np
from numpy.random import multivariate_normal
from typing import Optional, Callable
import matplotlib.pyplot as plt

class EnsembleKalmanFilter:
    def __init__(
        self,
        x:np.ndarray,                   # Initial state
        fx: Callable,                   # Nonlinear dynamics 
        N: int,                         # # of ensembles
        xdim: int,                      # Dimension for states
        zdim: int,                      # Dimension for measurements
        H: Optional[np.ndarray] = None, # Observation matrix
        Q: Optional[np.ndarray] = None, # Covariance of the process noise
        R: Optional[np.ndarray] = None, # Covariance of the measurement noise
        P: Optional[np.ndarray] = None, # Covariance of the state-estimate error
    ):

        self.N = N
        self.xdim = xdim
        self.zdim = zdim

        self.Q = np.eye(self.xdim) if Q is None else Q
        self.R = np.eye(self.zdim) if R is None else R

        self.x = x
        self.P = np.eye(self.xdim) if P is None else P
        self.Pxz = np.zeros((xdim, zdim), dtype = np.float32)
        self.Pzz = np.zeros((zdim, zdim), dtype = np.float32)

        self.H = H
        self.fx = fx

        self.K = np.zeros((xdim, zdim), dtype = np.float32) # Kalman gain matrix
        self.S = np.zeros((zdim, zdim), dtype = np.float32) # Innovation covariance
        
        self.ensemble_x = self.compute_diag_mvn_sample(x.reshape(-1,1), diag_cov=self.P, size = N)
        # self.ensemble_x = x.reshape(-1,1).repeat(N, axis = 1)    
        self.ensemble_z = np.zeros((zdim, N), dtype = np.float32)            # (zdim, N)

        self._mean_x = np.zeros(xdim)
        self._mean_z = np.zeros(zdim)
        
    def compute_diag_mvn_sample(self, mean:np.ndarray, diag_cov:np.ndarray, size:int):
        std_dev = np.sqrt(diag_cov)
        z = np.random.randn(mean.shape[0], size)
        return mean + std_dev @ z

    def update(self, z:Optional[np.ndarray]):
        
        if z is None:
            self.z = None
            return

        for idx in range(self.N):
            self.ensemble_z[:,idx] = (self.H @ self.ensemble_x[:,idx].reshape(-1,1)).ravel()

        self.z = z
        z_mean = np.mean(self.ensemble_z, axis = 1)

        A = self.ensemble_x - self.x.reshape(-1,1)
        HA = self.ensemble_z - z_mean.reshape(-1,1)
        
        self.Pzz = 1 / (self.N - 1) * HA@HA.T + self.R
        self.Pxz = 1 / (self.N - 1) * A@HA.T

        self.S = self.Pzz
        self.K = self.Pxz @ np.linalg.inv(self.S)

        err_z = self.compute_diag_mvn_sample(self._mean_z.reshape(-1,1), self.R, self.N)
        
        D = np.concatenate([self.z for _ in range(self.N)], axis = 1) + err_z
        Xp = self.ensemble_x + self.K@(D - self.ensemble_z)
        
        self.x = np.mean(Xp, axis = 1)
        self.P = self.P - self.K@self.S@self.K.T

    def predict(self):
        err_x = self.compute_diag_mvn_sample(self._mean_x.reshape(-1,1), self.Q, self.N)

        for idx in range(self.N):
            self.ensemble_x[:,idx] = self.fx(self.ensemble_x[:,idx]).ravel()
            
        self.ensemble_x += err_x
        self.x = np.mean(self.ensemble_x, axis = 1, keepdims=True)
        
        return self.x

if __name__ == "__main__":

    # setup
    dt = 0.01
    sig_a = 0.25
    sig_z = 0.25
    w = 1.0

    # Matrix for describing dynamics
    F = np.array([[1, dt], [0, 1]])
    G = np.array([0.5 * dt**2, dt]).reshape(-1, 1)
    Q = G @ G.T * sig_a**2
    H = np.array([1, 0]).reshape(1, 2)
    P = np.array([[sig_a**2, 0], [0, sig_a**2]])
    R = np.eye(1) * sig_z**2
    B = np.array([0.5 * dt**2, dt]).reshape(-1, 1)

    x0 = np.array([0.0, 4.0]).reshape(-1, 1)
    y_true = 0

    traj_estimate = []

    N_iter = 1000
    N = 64

    def f(u):
        a = np.array([-(w**2) * u]).reshape(1, 1)
        return a

    def fx(u:np.ndarray):
        return F@u.reshape(-1,1) + B@f(u[0])

    def m(t):
        return 4 * np.sin(w * t) + np.random.normal(0, sig_z)

    # Kalman Filter
    enkf = EnsembleKalmanFilter(x0, fx, N, 2, 1, H, Q, R, P)

    measurements = [m(t) for t in [i * dt for i in range(N_iter)]]

    for i in np.arange(0, N_iter):
        x = enkf.predict()
        y_pred = H @ x

        traj_estimate.append(y_pred.reshape(-1,1))

        y_true = measurements[i]
        enkf.update(np.array(y_true).reshape(-1,1))

    measurements = np.array(measurements)
    predictions = np.concatenate(traj_estimate, axis=1).T

    plt.plot(measurements, label="Measurements")
    plt.plot(predictions[:, 0], label="EnKF Prediction")
    plt.legend()
    plt.savefig("./results/test/test_enkf.png")