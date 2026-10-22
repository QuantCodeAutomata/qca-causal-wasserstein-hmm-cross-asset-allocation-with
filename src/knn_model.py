import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


class KNNMomentEstimator:
    """
    Strictly causal K-nearest-neighbor conditional moment estimator.
    Uses expanding history, standardized Euclidean distance.
    # Using sklearn NearestNeighbors — Context7 confirmed
    # Using sklearn LedoitWolf for covariance — Context7 confirmed
    """

    def __init__(self, k_nn: int = 50, scale: bool = True):
        self.k_nn = k_nn
        self.scale = scale

    def estimate(self, X_hist: np.ndarray, r_hist: np.ndarray,
                 x_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate conditional mean and covariance at x_query.

        X_hist: (T, D) feature matrix of past observations
        r_hist: (T, N) return matrix corresponding to X_hist
                r_hist[t] = asset returns for the day AFTER feature X_hist[t]
        x_query: (D,) current feature vector

        Returns:
          - mu: (N,) conditional mean
          - Sigma: (N, N) Ledoit-Wolf shrunk covariance

        Documented choices:
        - Distance metric: Euclidean on standardized features (scale=True)
        - K_NN = 50 neighbors
        - LedoitWolf shrinkage for covariance
        - Scaler fit on X_hist at each call (expanding history, causal)
        """
        T = len(X_hist)
        k = min(self.k_nn, T)

        # Scale features
        if self.scale and T > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_hist)
            x_q_scaled = scaler.transform(x_query.reshape(1, -1)).ravel()
        else:
            X_scaled = X_hist
            x_q_scaled = x_query

        # Find K nearest neighbors
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_scaled)
        indices = nn.kneighbors(x_q_scaled.reshape(1, -1), return_distance=False)[0]

        neighbor_returns = r_hist[indices]  # (K, N)

        # Conditional mean
        mu = neighbor_returns.mean(axis=0)

        # Ledoit-Wolf covariance
        if len(neighbor_returns) >= 2:
            lw = LedoitWolf()
            lw.fit(neighbor_returns)
            Sigma = lw.covariance_
        else:
            Sigma = np.eye(neighbor_returns.shape[1]) * 1e-4

        return mu, Sigma
