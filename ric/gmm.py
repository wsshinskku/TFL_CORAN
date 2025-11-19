# ric/gmm.py
# Lightweight GMM with EM; returns soft memberships γ_ik
from __future__ import annotations
from typing import Tuple
import numpy as np

class SoftGMM:
    def __init__(self, n_components: int=3, max_iter: int=100, tol: float=1e-3, reg_covar: float=1e-6, covariance_type: str="full", rng: int=42):
        assert covariance_type in ("full", "diag")
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg_covar
        self.covariance_type = covariance_type
        self.rng = np.random.RandomState(rng)
        self.pi = None
        self.mu = None
        self.Sigma = None

    def _init_params(self, Z: np.ndarray):
        N, D = Z.shape
        idx = self.rng.choice(N, size=self.K, replace=False)
        self.mu = Z[idx].copy()                          # K,D
        if self.covariance_type == "full":
            self.Sigma = np.stack([np.cov(Z.T) + self.reg*np.eye(D) for _ in range(self.K)], axis=0)   # K,D,D
        else:
            self.Sigma = np.stack([np.var(Z, axis=0) + self.reg for _ in range(self.K)], axis=0)       # K,D
        self.pi = np.ones(self.K, dtype=np.float64) / self.K

    def _log_gauss(self, Z: np.ndarray) -> np.ndarray:
        # returns log N(z|mu_k, Sigma_k) for all k: [N,K]
        N, D = Z.shape
        logp = np.zeros((N, self.K), dtype=np.float64)
        for k in range(self.K):
            diff = Z - self.mu[k]
            if self.covariance_type == "full":
                S = self.Sigma[k] + self.reg*np.eye(D)
                invS = np.linalg.inv(S)
                logdet = np.linalg.slogdet(S)[1]
                quad = np.sum(diff @ invS * diff, axis=1)
            else:
                var = self.Sigma[k] + self.reg
                logdet = np.sum(np.log(var))
                quad = np.sum((diff*diff)/var, axis=1)
            logp[:, k] = -0.5*(D*np.log(2*np.pi) + logdet + quad)
        return logp

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Fit on Z [N,D]; return responsibilities gamma [N,K].
        """
        Z = Z.astype(np.float64)
        N, D = Z.shape
        if self.mu is None:
            self._init_params(Z)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: γ_nk ∝ π_k N(z_n|μ_k, Σ_k)
            logp = self._log_gauss(Z) + np.log(self.pi + 1e-12)
            # log-sum-exp for stability
            m = logp.max(axis=1, keepdims=True)
            logp_norm = logp - m
            exp_logp = np.exp(logp_norm)
            denom = exp_logp.sum(axis=1, keepdims=True) + 1e-12
            gamma = exp_logp / denom                                            # [N,K]
            ll = np.sum(m + np.log(denom))                                      # log-likelihood

            # M-step
            Nk = gamma.sum(axis=0) + 1e-12                                      # [K]
            self.pi = Nk / N
            self.mu = (gamma.T @ Z) / Nk[:, None]                               # [K,D]
            if self.covariance_type == "full":
                Sigma = np.zeros((self.K, D, D), dtype=np.float64)
                for k in range(self.K):
                    diff = Z - self.mu[k]
                    Sigma[k] = (diff.T * gamma[:, k]) @ diff / Nk[k]
                    Sigma[k].flat[:: D+1] += self.reg                            # add reg to diag
                self.Sigma = Sigma
            else:
                # diag
                var = np.zeros((self.K, D), dtype=np.float64)
                for k in range(self.K):
                    diff = Z - self.mu[k]
                    var[k] = (gamma[:, k][:, None] * (diff*diff)).sum(axis=0) / Nk[k] + self.reg
                self.Sigma = var

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return gamma.astype(np.float32)
