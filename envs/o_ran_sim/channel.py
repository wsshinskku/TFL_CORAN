# envs/o_ran_sim/channel.py
# Channel/SINR/MCS/TB/Bler models (lightweight)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import math

@dataclass
class McsDecision:
    ue_id: str
    subband: int
    mcs_index: int
    sinr_db: float
    tb_bits: int
    prio: int = 1
    @property
    def priority(self) -> int: return self.prio
    @priority.setter
    def priority(self, v: int): self.prio = int(v)

class ChannelModel:
    """
    Simplified DL link budget so that SNR ranges ~[-5, 30] dB across cell area.
    - Pathloss: 3GPP-like log-distance (coarse)
    - Fading: log-normal shadowing + small Rayleigh
    - BLER: logistic vs (SINR - threshold_mcs)
    - TB size: table by MCS, per 1ms slot (scaled for 100 MHz carrier)
    """
    def __init__(self, center_freq_ghz: float, bandwidth_mhz: int, n_subbands: int,
                 reuse_allowed: bool, rng: np.random.RandomState):
        self.fc = center_freq_ghz
        self.bw = bandwidth_mhz
        self.n_subbands = n_subbands
        self.reuse_allowed = reuse_allowed
        self.rng = rng

        # MCS threshold table (dB) and TB bits per 1ms (scaled)
        self._mcs_snr_thresh = np.array([-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], dtype=np.float32)
        self._tb_bits = np.array([20e3, 28e3, 38e3, 50e3, 66e3, 86e3, 110e3, 140e3,
                                  180e3, 230e3, 290e3, 360e3, 450e3, 560e3, 700e3, 880e3], dtype=np.float32).astype(int)

        # Link budget constants (tuned for reasonable SNR span)
        self._tx_power_dbm = 46.0   # gNB DL (approx)
        self._tx_ant_gain_db = 15.0
        self._rx_ant_gain_db = 0.0
        self._noise_dbm = -174 + 10*np.log10(self.bw*1e6) + 7.0  # -174 dBm/Hz + BW + NF(~7dB)

    def num_mcs(self) -> int:
        return len(self._mcs_snr_thresh)

    def tb_bits_for_mcs(self, mcs_index: int) -> int:
        return int(self._tb_bits[int(np.clip(mcs_index, 0, self.num_mcs()-1))])

    # ------------- prediction APIs -------------
    def predict_link(self, ue_xy, gnb_xy, mcs_index: int, subband: int) -> McsDecision:
        sinr_db = self.predict_sinr_only(ue_xy, gnb_xy)
        tb_bits = self.tb_bits_for_mcs(mcs_index)
        return McsDecision(ue_id="", subband=int(subband), mcs_index=int(mcs_index),
                           sinr_db=float(sinr_db), tb_bits=tb_bits)

    def predict_sinr_only(self, ue_xy, gnb_xy) -> float:
        d = np.linalg.norm(ue_xy - gnb_xy) + 1.0
        # 3GPP-like coarse PL: 32.4 + 21log10(fc_GHz) + 31.9log10(d_m)
        pl_db = 32.4 + 21*np.log10(self.fc) + 31.9*np.log10(d)
        # Shadowing + small-scale fading
        sh_db = self.rng.normal(loc=0.0, scale=6.0)  # log-normal shadowing (σ≈6dB)
        fad_db = 10*np.log10(max(self.rng.rayleigh(scale=1.0), 1e-3))
        rx_dbm = self._tx_power_dbm + self._tx_ant_gain_db + self._rx_ant_gain_db - pl_db + sh_db + fad_db
        sinr_db = rx_dbm - self._noise_dbm
        return float(sinr_db)

    def bler_from_sinr(self, mcs_index: int, sinr_db: float) -> float:
        # Logistic BLER around threshold; steeper at higher MCS
        thr = self._mcs_snr_thresh[int(np.clip(mcs_index, 0, self.num_mcs()-1))]
        k = 1.2 + 0.05 * mcs_index
        x = k * (sinr_db - thr)
        bler = 1.0 / (1.0 + np.exp(x))
        return float(np.clip(bler, 0.0, 1.0))

    # ------------- proxies for RIC features S/I/T/M -------------
    def rx_power_proxy(self, ue_xy, gnb_xy) -> float:
        # map received power (dBm) to [0,1] via softmin at -100..-60 dBm
        sinr = self.predict_sinr_only(ue_xy, gnb_xy)
        # reuse sinr as proxy for S (monotonic with rx power)
        return float(np.clip((sinr + 10.0) / 40.0, 0.0, 1.0))
