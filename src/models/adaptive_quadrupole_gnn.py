"""Adaptive Quadrupole Graph Neural Network (AQ-GNN) for Crystal Max-EFG Prediction.

Mathematical Foundations & Physics:
1. Physical Quadrupole Tensor Field & Legendre Polynomials:
   V_ij = (1 / 4*pi*eps_0) * sum_k q_k * (3 * r_ik (x) r_ik - r_ik^2 * I) / r_ik^5
   Tr(T_ij * T_ik) = 6 * P_2(cos theta_jik), evaluating angular 3-body coordination
   without explicit line-graph memory explosion.
2. Analytic Invariant Decomposition of Local Quadrupole Environment:
   The rank-2 traceless quadrupole tensor Q_i has two fundamental algebraic invariants:
   - Scale (Frobenius norm): ||Q_i||_F
   - Shape/Asymmetry angle: cos(3*theta) = 3*sqrt(6)*det(Q_i) / ||Q_i||_F^3
   Naturally vanishes on cubic/octahedral sites (exact symmetry cancellation).
3. Inversion-Asymmetry Dipole Invariant:
   D_i = sum_j w_ij * r_hat_ij / r_ij^2 measures local polar distortion.
4. Physical Sternheimer Antishielding Amplification:
   The nuclear EFG is amplified by core electron distortion (1 - gamma_inf),
   modeled via elemental priors (Z, mass, electronegativity, covalent radius)
   and a learned site-level amplification factor sigma_i.
5. Differentiable Smooth-Max / Softmax-Weighted Extreme Pooling:
   Models the generative extreme-value nature: y = max_s V_site(s),
   replacing harmful global mean pooling with a learnable temperature-scaled
   log-sum-exp operator:
       y_smooth = v_max + tau * log(sum_s exp((v_s - v_max) / tau))
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.features.descriptors import ELEMENT_PROPERTIES
from src.features.graph import ELEMENT_TO_Z


class GaussianSmearing(nn.Module):
    """Radial basis functions with distance cutoff."""
    def __init__(self, start: float = 0.0, stop: float = 6.0, num_gaussians: int = 40):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function vanishing at r_cut."""
    def __init__(self, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        cut = 0.5 * (torch.cos(dist * (math.pi / self.cutoff)) + 1.0)
        return torch.where(dist < self.cutoff, cut, torch.zeros_like(cut))


class AdaptiveQuadrupoleGNN(nn.Module):
    """Adaptive Quadrupole GNN specifically tailored for electric field gradient physics."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 40,
        radius_cutoff: float = 6.0,
        num_quad_channels: int = 4,
        dropout: float = 0.05,
        pooling: str = "smoothmax"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.radius_cutoff = radius_cutoff
        self.num_quad_channels = num_quad_channels
        self.pooling = pooling

        # 1. Elemental & Physical Prior Embeddings
        self.atom_embed = nn.Embedding(119, hidden_dim)

        # Build normalized physical properties table [119, 6]
        phys_table = np.zeros((119, 6), dtype=np.float32)
        for el, props in ELEMENT_PROPERTIES.items():
            z = ELEMENT_TO_Z.get(el, 0)
            if 0 < z < 119:
                phys_table[z] = props[:6]
        # Standardize non-zero entries
        mask = phys_table.sum(axis=-1) > 0
        mean = phys_table[mask].mean(axis=0, keepdims=True)
        std = phys_table[mask].std(axis=0, keepdims=True) + 1e-6
        phys_table[mask] = (phys_table[mask] - mean) / std

        self.register_buffer("phys_properties", torch.tensor(phys_table, dtype=torch.float32))
        self.phys_proj = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 2. Radial & Multipole Edge Features
        self.rbf = GaussianSmearing(0.0, radius_cutoff, num_gaussians)
        self.cutoff = CosineCutoff(radius_cutoff)

        # Edge feature dim: num_gaussians + 5 power law terms (1/r, 1/r^2, 1/r^3, 1/r^4, 1/r^5)
        edge_raw_dim = num_gaussians + 5
        self.edge_init = nn.Sequential(
            nn.Linear(edge_raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Quadrupole edge projection weights
        self.quad_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_quad_channels),
            nn.Sigmoid()
        )
        # Dipole edge projection weights
        self.dipole_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_quad_channels),
            nn.Sigmoid()
        )

        # Geometric invariants projection:
        # Per channel: Frob_norm, Det, Shape_psi, Dipole_norm -> 4 * num_quad_channels
        geom_dim = 4 * num_quad_channels
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 3. Message Passing Layers
        self.message_mlps = nn.ModuleList()
        self.gate_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.message_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.gate_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
                nn.Sigmoid()
            ))
            self.node_mlps.append(nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.node_norms.append(nn.LayerNorm(hidden_dim))
            self.edge_norms.append(nn.LayerNorm(hidden_dim))

        # 4. Site-Level EFG Predictor Head with Sternheimer Antishielding Factor
        self.site_val_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.sternheimer_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.site_lat_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # 5. Global Crystal Correction & Attention
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.crystal_correction = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Learnable Smooth-Max temperature parameter (initialized to log(4.0))
        self.log_tau = nn.Parameter(torch.tensor([math.log(4.0)]))

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
        - y_crystal: [B] predicted crystal maximum EFG scalar
        - v_sites: [N] predicted sitewise EFG scalar proxy
        """
        # 1. Node Initialization
        x_idx = torch.clamp(batch.x, 0, 118)
        h = self.atom_embed(x_idx) + self.phys_proj(self.phys_properties[x_idx])

        # 2. Edge Features
        dist = batch.edge_dist
        f_cut = self.cutoff(dist).unsqueeze(-1)

        # Radial Gaussians + 1/r, 1/r^2, 1/r^3, 1/r^4, 1/r^5 multipole power-law terms
        rbf_feat = self.rbf(dist) * f_cut
        r_safe = torch.clamp(dist, min=0.8).unsqueeze(-1)
        power_feat = torch.cat([
            1.0 / r_safe,
            1.0 / (r_safe ** 2),
            1.0 / (r_safe ** 3),
            1.0 / (r_safe ** 4),
            1.0 / (r_safe ** 5)
        ], dim=-1) * f_cut

        raw_edge = torch.cat([rbf_feat, power_feat], dim=-1)
        e = self.edge_init(raw_edge)

        # 3. Geometric Quadrupole & Dipole Coordination Tensors
        # Unit displacement vectors: u_ij = r_ij / ||r_ij||
        u = batch.edge_vec / (dist.unsqueeze(-1) + 1e-8)  # [E, 3]
        u_outer = torch.bmm(u.unsqueeze(-1), u.unsqueeze(-2))  # [E, 3, 3]
        eye = torch.eye(3, device=dist.device).unsqueeze(0)  # [1, 3, 3]
        T_mat = 3.0 * u_outer - eye  # [E, 3, 3]

        # Multi-channel quadrupole weights: decay as 1/r^3
        q_weights = self.quad_weights(e) / (r_safe ** 3) * f_cut  # [E, K_quad]
        # Multi-channel dipole weights: decay as 1/r^2
        d_weights = self.dipole_weights(e) / (r_safe ** 2) * f_cut  # [E, K_quad]

        src, dst = batch.edge_index
        N = h.size(0)

        frob_invariants = []
        det_invariants = []
        shape_invariants = []
        dipole_invariants = []

        for k in range(self.num_quad_channels):
            # Quadrupole tensor for channel k
            w_qk = q_weights[:, k].unsqueeze(-1).unsqueeze(-1)
            T_k = T_mat * w_qk
            Q_geom = torch.zeros(N, 3, 3, device=dist.device)
            Q_geom.index_add_(0, dst, T_k)

            # Dipole vector for channel k
            w_dk = d_weights[:, k].unsqueeze(-1)
            D_vec = torch.zeros(N, 3, device=dist.device)
            D_vec.index_add_(0, dst, u * w_dk)

            # Compute analytic tensor invariants
            frob_k = torch.norm(Q_geom, dim=(-2, -1)).unsqueeze(-1)  # [N, 1]
            det_k = torch.linalg.det(Q_geom).unsqueeze(-1)  # [N, 1]
            psi_k = torch.clamp(3.0 * math.sqrt(6.0) * det_k / (frob_k ** 3 + 1e-6), -1.0, 1.0)  # [N, 1]
            dip_k = torch.norm(D_vec, dim=-1, keepdim=True)  # [N, 1]

            frob_invariants.append(frob_k)
            det_invariants.append(det_k)
            shape_invariants.append(psi_k)
            dipole_invariants.append(dip_k)

        all_geom = torch.cat(frob_invariants + det_invariants + shape_invariants + dipole_invariants, dim=-1)
        geom_node_feat = self.geom_proj(all_geom)  # [N, hidden_dim]

        # 4. Gated Message Passing with Residual Updates
        for layer_idx in range(self.num_layers):
            edge_cat = torch.cat([h[src], h[dst], e], dim=-1)
            msg = self.message_mlps[layer_idx](edge_cat) * self.gate_mlps[layer_idx](edge_cat)

            agg_msg = torch.zeros_like(h)
            agg_msg.index_add_(0, dst, msg)

            node_in = torch.cat([h, agg_msg, geom_node_feat], dim=-1)
            h = h + self.node_norms[layer_idx](self.node_mlps[layer_idx](node_in))

            edge_in = torch.cat([e, h[src], h[dst]], dim=-1)
            e = e + self.edge_norms[layer_idx](self.edge_mlps[layer_idx](edge_in))

        # 5. Site-Level Physical EFG Prediction
        v_val = self.site_val_head(h).squeeze(-1)  # [N]
        sigma = 1.0 + self.sternheimer_head(h).squeeze(-1)  # [N]
        v_lat = self.site_lat_head(geom_node_feat).squeeze(-1)  # [N]

        # Nuclear EFG decomposition: amplified valence + lattice
        v_sites = sigma * v_val + v_lat  # [N]

        # 6. Adaptive Smooth-Maximum Crystal Pooling
        tau = torch.exp(self.log_tau).clamp(min=0.5, max=20.0)

        from torch_geometric.nn import global_max_pool
        from torch_geometric.utils import softmax
        max_v = global_max_pool(v_sites, batch.batch)  # [B]

        if self.pooling == "boltzmann":
            # Boltzmann Softmax-Weighted Extreme Value Pooling:
            # Strictly convex combination: min(v) <= y_pool <= max(v)
            # Immune to artificial cell size inflation (+tau*log(N))
            w_sites = softmax(v_sites / tau, batch.batch)
            y_pool = torch.zeros(max_v.size(0), device=h.device)
            y_pool.index_add_(0, batch.batch, w_sites * v_sites)
        else:
            # Smooth-max (Log-Sum-Exp)
            v_scaled = (v_sites - max_v[batch.batch]) / tau
            exp_v = torch.exp(v_scaled)
            sum_exp = torch.zeros(max_v.size(0), device=h.device)
            sum_exp.index_add_(0, batch.batch, exp_v)
            y_pool = max_v + tau * torch.log(sum_exp.clamp(min=1e-8))  # [B]

        # Attention-weighted crystal context correction
        att_scores = softmax(self.attention_w(h).squeeze(-1), batch.batch)
        att_h = torch.zeros(max_v.size(0), h.size(-1), device=h.device)
        att_h.index_add_(0, batch.batch, h * att_scores.unsqueeze(-1))
        delta_y = self.crystal_correction(att_h).squeeze(-1)

        y_crystal = F.softplus(y_pool + delta_y)

        return y_crystal, v_sites
