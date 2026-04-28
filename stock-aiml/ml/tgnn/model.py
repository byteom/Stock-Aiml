"""TGNN: Temporal Graph Neural Network for cross-asset contagion modeling.

Architecture:
  - Node = stock/index
  - Node features = sequence of per-stock features over a rolling window
  - Temporal encoder: GRU (or TCN) maps feature sequences → node embeddings
  - Graph attention layers learn time-varying edge weights (cross-stock relationships)
  - Output: per-node contextualized embedding + edge weights (contagion signals)

For the MVP (single NIFTY 50 index):
  - Treat the index as a single node with a self-loop
  - The model still runs end-to-end; when multi-stock data is provided,
    it automatically uses the full graph attention architecture
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset


# ─── Attention-based graph layer ───────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """Single graph attention layer (GAT-style)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim   = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.bias = nn.Parameter(torch.empty(num_heads, 1))

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:   (B, N, D)  node features
            adj: (B, N, N)  adjacency matrix (soft mask); if None → dense attention

        Returns:
            (B, N, out_dim) updated node embeddings
        """
        B, N, _ = x.shape

        # Project and split into heads
        x_proj = self.W(x)                              # (B, N, out_dim)
        x_heads = x_proj.view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D')
        x_i = x_heads.unsqueeze(2)                       # (B, N, 1, H, D')
        x_j = x_heads.unsqueeze(1)                       # (B, 1, N, H, D')

        # Concatenate for attention
        pair = torch.cat([x_i, x_j], dim=-1)            # (B, N, N, H, 2*D')
        pair_flat = pair.view(B, N, N, self.num_heads, 2 * self.head_dim)

        # Attention scores
        att = torch.einsum("bnhd,hd->bnh", pair_flat, self.att) + self.bias
        att = F.leaky_relu(att, 0.2)

        # Masking (if adjacency provided)
        if adj is not None:
            mask = (adj + torch.eye(N, device=x.device) * 0.5).clamp(0, 1)  # self-loops always on
            att = att.masked_fill(mask == 0, -1e9)

        att_norm = F.softmax(att, dim=2)                 # (B, N, N, H)
        att_norm = self.dropout(att_norm)

        # Apply attention to values
        out = torch.einsum("bnmh,bmhd->bnhd", att_norm, x_heads.permute(1, 2, 0, 3))
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return out


# ─── Temporal encoder (GRU) ───────────────────────────────────────────────────

class TemporalEncoderGRU(nn.Module):
    """Encodes a feature sequence per node using a GRU."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — T is the sequence window length

        Returns:
            (B, hidden_dim) — final hidden state as node embedding
        """
        out, h = self.gru(x)
        # Use last layer, last time-step
        return self.proj(h[-1])                          # (B, hidden_dim)


class TemporalEncoderTCN(nn.Module):
    """Temporal Convolutional Network encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_d, hidden_dim, kernel_size,
                                     padding=dilation * (kernel_size - 1), dilation=dilation))
            if i != num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        out = self.net(x)
        # Take last timestep
        return self.proj(out[:, :, -1])


# ─── Full TGNN model ────────────────────────────────────────────────────────────

class TGNN(nn.Module):
    """
    Temporal Graph Neural Network for multi-stock contagion modeling.

    Forward pass:
      1. Temporal encoder: feature sequence → per-node embedding
      2. Stack of graph attention layers: propagate cross-stock signals
      3. Optional final readout: global graph embedding (mean pool)
      4. Output: node embeddings + edge weight logits (contagion scores)
    """

    def __init__(
        self,
        num_nodes:           int,       # number of stocks (nodes)
        node_feature_dim:    int,       # features per node per timestep
        embedding_dim:       int = 64,
        hidden_dim:          int = 128,
        num_layers:          int = 3,
        num_heads:           int = 4,
        temporal_encoder:    str = "gru",  # "gru" | "tcn"
        dropout:             float = 0.1,
        use_edge_learning:   bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.use_edge_learning = use_edge_learning

        # Temporal encoder per node
        TemporalEnc = TemporalEncoderGRU if temporal_encoder == "gru" else TemporalEncoderTCN
        self.temp_enc = TemporalEnc(node_feature_dim, hidden_dim, num_layers=2, dropout=dropout)

        # Learnable initial edge weights (Granger-like baseline)
        self.edge_baseline = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.edge_baseline.data * 0.1)

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, embedding_dim, num_heads=num_heads, dropout=dropout)
            )
            in_dim = embedding_dim

        # Final node embedding projector
        self.node_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Link prediction head (for contagion / edge weight output)
        if use_edge_learning:
            self.edge_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )

        # Return prediction head (supervised pretraining)
        self.return_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, N, D) — batch of node feature sequences
        adj: torch.Tensor | None = None,  # (N, N) or (B, N, N) optional prior
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:   (B, T, N, D)  — B batches, T timesteps, N nodes, D features
            adj: (N, N) prior adjacency (e.g., correlation matrix)

        Returns:
            dict with:
              node_embeddings: (B, N, embedding_dim)
              edge_weights:    (B, N, N)  contagion scores
              return_pred:     (B, N)     next-step return prediction
        """
        B, T, N, D = x.shape

        # Reshape to (B*T, N, D) for temporal encoding
        x_flat = x.view(B * T, N, D)

        # Temporal encode each node
        # Process per-node: (B*T, N, D) → GRU takes (B*T, T_node, D) — but we have no T_node dim
        # Instead: reshape per node to (B, T, D) per node, process, then stack
        node_emb_list = []
        for node_i in range(N):
            seq = x[:, :, node_i, :]            # (B, T, D)
            emb = self.temp_enc(seq)              # (B, hidden_dim)
            node_emb_list.append(emb)

        node_embs = torch.stack(node_emb_list, dim=1)  # (B, N, hidden_dim)

        # Graph attention layers
        for gat in self.gat_layers:
            prior = self.edge_baseline.unsqueeze(0).expand(B, -1, -1)
            node_embs = gat(node_embs, adj=prior) + node_embs  # residual

        # Final projection
        node_embs = self.node_proj(node_embs)    # (B, N, embedding_dim)

        # Edge weights (contagion scores)
        if self.use_edge_learning:
            edge_w = torch.zeros(B, N, N, device=x.device)
            for i in range(N):
                for j in range(N):
                    emb_pair = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                    edge_w[:, i, j] = self.edge_predictor(emb_pair).squeeze(-1)
            # Softmax along j dimension for interpretability
            edge_weights = F.softmax(edge_w, dim=-1)
        else:
            edge_weights = (self.edge_baseline.softmax(dim=-1)).unsqueeze(0).expand(B, -1, -1)

        # Return prediction (supervised pretraining objective)
        return_pred = self.return_head(node_embs).squeeze(-1)   # (B, N)

        return {
            "node_embeddings": node_embs,
            "edge_weights":   edge_weights,
            "return_pred":     return_pred,
        }

    def get_contagion_matrix(self, node_embs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise contagion scores from node embeddings."""
        N = node_embs.shape[1]
        scores = torch.zeros(N, N, device=node_embs.device)
        for i in range(N):
            for j in range(N):
                emb_pair = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                scores[i, j] = self.edge_predictor(emb_pair).squeeze(-1)
        return F.softmax(scores, dim=-1)


# ─── Data loader for TGNN ─────────────────────────────────────────────────────

class TGNNDataLoader:
    """
    Converts a multi-stock DataFrame dict into PyTorch tensors for TGNN training.

    Input:
        multi_df: dict[str, pd.DataFrame] — stock_symbol -> OHLCV+features df
        window:   int — number of historical bars per sample
        horizon:   int — prediction horizon (bars ahead to predict return)

    Output:
        X: (B, T, N, D)  node features
        y: (B, N)        next-step returns (labels)
        adj_matrices: (B, N, N)  correlation-based priors
    """

    def __init__(self, multi_df: dict[str, pd.DataFrame], window: int = 20, horizon: int = 1):
        self.multi_df = multi_df
        self.symbols  = list(multi_df.keys())
        self.N        = len(self.symbols)
        self.window   = window
        self.horizon  = horizon
        self._df      = self._align_and_concat()

    def _align_and_concat(self) -> pd.DataFrame:
        dfs = {}
        for sym, df in self.multi_df.items():
            df = df.set_index("timestamp").sort_index()
            dfs[sym] = df[["close", "volume", "return_1d", "volatility"]].rename(
                columns={c: f"{sym}_{c}" for c in df.columns}
            )
        aligned = pd.concat(dfs.values(), axis=1)
        return aligned.sort_index()

    def _build_adjacency(self, window_df: pd.DataFrame) -> np.ndarray:
        """Correlation-based adjacency from a window of data."""
        returns_cols = [c for c in window_df.columns if "return_1d" in c]
        if len(returns_cols) < 2:
            return np.ones((self.N, self.N)) / self.N
        ret_matrix = window_df[returns_cols].dropna().values.T
        if ret_matrix.shape[0] < 2:
            return np.ones((self.N, self.N)) / self.N
        corr = np.corrcoef(ret_matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        # Softmax-style: positive, normalized
        adj = np.maximum(corr, 0)
        adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-9)
        return adj

    def __iter__(self):
        df = self._df
        min_len = self.window + self.horizon + 5
        if len(df) < min_len:
            return

        for start in range(0, len(df) - min_len, max(1, (len(df) - min_len) // 500)):
            end_feat = start + self.window
            end_ret  = end_feat + self.horizon

            if end_ret > len(df):
                break

            window_df = df.iloc[start:end_feat]
            next_df   = df.iloc[end_feat:end_ret]

            # Build feature matrix (B, T, N, D)
            feature_cols = [c for c in df.columns if any(
                s in c for s in self.symbols
            )]
            feat_data = window_df[feature_cols].values.reshape(self.window, self.N, -1)
            D = feat_data.shape[-1]
            X = feat_data.reshape(1, self.window, self.N, D).astype(np.float32)

            # Build label: next-step return per stock
            y = []
            for sym in self.symbols:
                close_col = f"{sym}_close"
                if close_col in next_df.columns and f"{sym}_close" in window_df.columns:
                    ret = np.log(
                        next_df[close_col].iloc[-1] / window_df[close_col].iloc[-1]
                    )
                else:
                    ret = 0.0
                y.append(ret)
            y = np.array(y, dtype=np.float32)

            # Adjacency
            adj = self._build_adjacency(window_df).astype(np.float32)

            yield (
                torch.from_numpy(X),
                torch.from_numpy(y).unsqueeze(0),
                torch.from_numpy(adj).unsqueeze(0),
            )

    def __len__(self) -> int:
        return sum(1 for _ in self)


# ─── Trainer ─────────────────────────────────────────────────────────────────

class TGNNTrainer:
    """Trainer for the TGNN model with supervised + stability losses."""

    def __init__(
        self,
        model: TGNN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else torch.device(device)
        self.model  = model.to(self.device)
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience=5)

    def train_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.train()
        X, y = X.to(self.device), y.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)

        self.opt.zero_grad()
        out = self.model(X, adj=adj)

        # Supervised loss: predict next-step returns
        loss_sup = F.mse_loss(out["return_pred"], y)

        # Edge stability loss: encourage smooth changes in edge weights
        if self.model.use_edge_learning:
            edge_w = out["edge_weights"]
            # Penalize if edges are too uniform (encourage sparsity)
            entropy_loss = -(edge_w * (edge_w + 1e-9).log()).sum(-1).mean()
            stability = 0.05 * entropy_loss
        else:
            stability = torch.tensor(0.0, device=self.device)

        loss = loss_sup + stability
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return {
            "loss_total": loss.item(),
            "loss_sup":   loss_sup.item(),
            "loss_stability": stability.item(),
        }

    def eval_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            X, y = X.to(self.device), y.to(self.device)
            if adj is not None:
                adj = adj.to(self.device)
            out  = self.model(X, adj=adj)
            loss = F.mse_loss(out["return_pred"], y).item()
        return {"val_loss": loss, "return_pred": out["return_pred"].cpu()}

    def fit(
        self,
        dataloader: TGNNDataLoader,
        epochs: int = 50,
        val_split: float = 0.2,
        save_path: str | Path = "models/tgnn/best_model.pt",
    ) -> dict[str, list]:
        import os
        os.makedirs(os.path.dirname(save_path) if save_path else ".", exist_ok=True)

        history = {"epoch": [], "train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(epochs):
            train_losses, val_losses = [], []
            iter_loader = list(dataloader)

            for X, y, adj in iter_loader[: int(len(iter_loader) * (1 - val_split))]:
                metrics = self.train_step(X, y, adj)
                train_losses.append(metrics["loss_total"])

            for X, y, adj in iter_loader[int(len(iter_loader) * (1 - val_split)):]:
                metrics = self.eval_step(X, y, adj)
                val_losses.append(metrics["val_loss"])

            avg_train = np.mean(train_losses) if train_losses else float("inf")
            avg_val   = np.mean(val_losses)   if val_losses   else float("inf")
            self.sched.step(avg_val)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                self.save(save_path)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} — train: {avg_train:.5f}  val: {avg_val:.5f}")

        return history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "model_config": {
                "num_nodes": self.model.num_nodes,
                "embedding_dim": self.model.embedding_dim,
                "use_edge_learning": self.model.use_edge_learning,
            }
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
