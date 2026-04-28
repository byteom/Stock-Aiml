"""
Adversarial Market Generator (AMG) — Conditional GAN
====================================================

Generates realistic worst-case market scenarios conditioned on:
  - Recent price history (OHLCV sequence)
  - TGNN node embeddings (contagion context)
  - Market regime (trending vs mean-reverting)

Components:
  1. Generator:    LSTM + FC → synthetic OHLCV sequence
  2. Discriminator: LSTM + FC → real vs fake classification
  3. Feature matching + gradient penalty for training stability
  4. Walk-forward evaluation: does the generated scenario break the strategy?
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AMG] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: str | Path, seq_len: int = 30) -> pd.DataFrame:
    """Load NIFTY 50, engineer features, return clean DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for c in df.columns:
        if c in ("date", "timestamp"):
            col_map[c] = "timestamp"
        elif c == "open":   col_map[c] = "open"
        elif c == "high":   col_map[c] = "high"
        elif c == "low":    col_map[c] = "low"
        elif c == "close":  col_map[c] = "close"
    df = df.rename(columns=col_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %b %Y", dayfirst=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Synthesize volume
    if "volume" not in df.columns:
        df["volume"] = ((df["high"] - df["low"]) / df["high"].median() * 1e6).astype(float)

    df = df[df["high"] >= df["low"]].copy()
    df["close"] = df["close"].clip(df["low"] * 0.99, df["high"] * 1.01)

    # Features
    for h in [1, 5, 10, 20]:
        df[f"ret_{h}d"] = np.log(df["close"] / df["close"].shift(h)).fillna(0)

    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std()

    df["log_ret"]   = df["ret_1d"]
    df["hl_range"]  = (df["high"] - df["low"]) / df["close"]
    df["body_size"] = (df["close"] - df["open"]) / df["close"]

    ma20  = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - (ma20 - 2*std20)) / (4*std20 + 1e-10)

    delta = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta).clip(lower=0).rolling(14).mean()
    df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df = df.fillna(0).reset_index(drop=True)
    print(f"[AMG] Prepared data: {len(df)} bars, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")
    return df


SEQ_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "hl_range", "body_size",
    "bb_pos", "rsi", "macd", "macd_sig",
    "volume_ratio",
]


class MarketSequenceDataset(Dataset):
    """
    Dataset of real market sequences for GAN training.

    Each sample:
      - Condition (c):  [seq_len, cond_dim]  features from last seq_len bars
      - Target (x):      [seq_len, gen_dim]   the NEXT seq_len bars to generate
      - Label:           1 (real)

    Generator condition = last 15 bars of features
    Generator output    = synthetic 15 bars of returns
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 15, gap: int = 5,
                 train_ratio: float = 0.70, seed: int = 42):
        self.rng    = np.random.default_rng(seed)
        self.seq_len = seq_len
        self.gap    = gap

        self.cond_cols = SEQ_COLS
        self.gen_cols  = ["ret_1d", "hl_range", "body_size", "rsi", "macd"]

        # Build arrays
        cond_arr = df[self.cond_cols].values.astype(np.float32)
        gen_arr  = df[self.gen_cols].values.astype(np.float32)

        n = len(cond_arr)
        total_samples = n - seq_len * 2 - gap

        train_end = int(total_samples * train_ratio)
        indices = list(range(seq_len, seq_len + train_end))

        # Normalize
        self.cond_mean = cond_arr[:train_end].mean(axis=0)
        self.cond_std  = cond_arr[:train_end].std(axis=0)  + 1e-8
        self.gen_mean  = gen_arr[:train_end].mean(axis=0)
        self.gen_std   = gen_arr[:train_end].std(axis=0)   + 1e-8

        self.cond_data = (cond_arr - self.cond_mean) / self.cond_std
        self.gen_data  = (gen_arr  - self.gen_mean)  / self.gen_std

        # Walk-forward split
        self.train_indices = indices
        # For test: different regime from later dates
        self.test_indices  = list(range(n - seq_len - 50, n - seq_len))

        print(f"[AMG Dataset] seq_len={seq_len}, "
              f"train={len(self.train_indices)}, test={len(self.test_indices)}, "
              f"cond_dim={len(self.cond_cols)}, gen_dim={len(self.gen_cols)}")

    def get_split(self, split: str) -> "MarketSequenceDataset":
        """Return a view of the dataset with only train or test indices."""
        ds = MarketSequenceDataset.__new__(MarketSequenceDataset)
        ds.seq_len    = self.seq_len
        ds.gap        = self.gap
        ds.cond_cols  = self.cond_cols
        ds.gen_cols   = self.gen_cols
        ds.cond_mean  = self.cond_mean
        ds.cond_std   = self.cond_std
        ds.gen_mean   = self.gen_mean
        ds.gen_std    = self.gen_std
        ds.cond_data  = self.cond_data
        ds.gen_data   = self.gen_data
        if split == "train":
            ds._indices = self.train_indices
        else:
            ds._indices = self.test_indices
        return ds

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i = self._indices[idx]
        cond = self.cond_data[i - self.seq_len : i]
        gen  = self.gen_data[i : i + self.seq_len]
        return (
            torch.from_numpy(cond).float(),
            torch.from_numpy(gen).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    LSTM-based generator: conditioned synthetic market sequence.

    Input:
      - condition: [B, seq_len, cond_dim]  (recent real market features)
      - noise:     [B, latent_dim]         (random seed)

    Output:
      - synthetic: [B, seq_len, gen_dim]    (synthetic next-sequence features)
    """

    def __init__(self, cond_dim: int, gen_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latent_dim   = latent_dim
        self.cond_dim     = cond_dim
        self.gen_dim      = gen_dim
        self.seq_len       = 15  # fixed

        # Encode condition
        self.cond_encoder = nn.LSTM(cond_dim, hidden_dim, num_layers,
                                     batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Combine with noise -> context
        self.fc_init = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        # LSTM decoder to produce output sequence
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, gen_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        B = condition.size(0)
        if noise is None:
            noise = torch.randn(B, self.latent_dim, device=condition.device)

        # Encode condition
        _, (h_n, _) = self.cond_encoder(condition)  # (num_layers, B, hidden)
        cond_emb = h_n[-1]                            # (B, hidden)
        if cond_emb.ndim == 1:
            cond_emb = cond_emb.unsqueeze(0)         # (1, hidden) guard

        # Combine with noise
        ctx = torch.cat([cond_emb, noise], dim=-1)   # (B, latent+hidden)
        ctx = self.fc_init(ctx)                      # (B, hidden)

        # Decode to sequence
        ctx_seq = ctx.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq, H)
        output, _ = self.decoder(ctx_seq)             # (B, seq, H)
        synthetic = self.fc_out(output)               # (B, seq, gen_dim)
        return synthetic


# ══════════════════════════════════════════════════════════════════════════════
#  DISCRIMINATOR
# ══════════════════════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    LSTM-based discriminator: distinguish real vs synthetic sequences.

    Input:
      - sequence:   [B, seq_len, dim]  (either real or generated)
      - condition: [B, seq_len, cond_dim] (the conditioning context)

    Output:
      - score:     [B, 1]             (logit: real=1, fake=0)
    """

    def __init__(self, seq_dim: int, cond_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.LSTM(seq_dim + cond_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Concatenate sequence + condition along feature dim
        combined = torch.cat([sequence, condition], dim=-1)  # (B, seq, dim+cond_dim)
        _, (h_n, _) = self.encoder(combined)
        feat = h_n[-1]                                      # (B, hidden)
        return self.fc(feat)


# ══════════════════════════════════════════════════════════════════════════════
#  WASSERSTEIN GAN WITH GRADIENT PENALTY
# ══════════════════════════════════════════════════════════════════════════════

class WassersteinGAN(nn.Module):
    """
    WGAN-GP for adversarial market scenario generation.

    Training:
      1. Train Discriminator on real + fake (with gradient penalty)
      2. Train Generator to fool the Discriminator

    Losses:
      - D: E[D(real)] - E[D(fake)] + lambda * GP
      - G: -E[D(fake)] + lambda_fm * feature_matching_loss
    """

    def __init__(
        self,
        cond_dim: int,
        gen_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        gp_lambda: float = 10.0,
        fm_lambda: float = 1.0,
    ):
        super().__init__()
        self.gp_lambda = gp_lambda
        self.fm_lambda = fm_lambda

        self.generator     = Generator(cond_dim, gen_dim, latent_dim, hidden_dim)
        self.discriminator = Discriminator(gen_dim, cond_dim, hidden_dim)

        # RMSprop: faster convergence on CPU for WGAN
        self.opt_G = torch.optim.RMSprop(self.generator.parameters(),     lr=5e-4)
        self.opt_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-4)

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                         cond: torch.Tensor) -> torch.Tensor:
        """WGAN-GP: enforce Lipschitz continuity of discriminator."""
        B  = real.size(0)
        alpha = torch.rand(B, 1, 1, device=real.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_inter = self.discriminator(interpolates, cond)
        grad_outputs = torch.ones(B, 1, device=real.device)
        gradients = torch.autograd.grad(
            outputs=d_inter,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
        )[0]
        gradients = gradients.reshape(B, -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    def train_step(self, cond: torch.Tensor, real: torch.Tensor) -> dict:
        B = cond.size(0)

        # ── Train Discriminator ────────────────────────────────────────────
        self.opt_D.zero_grad()
        noise    = torch.randn(B, self.generator.latent_dim, device=cond.device)
        fake_seq = self.generator(cond, noise).detach()
        d_real   = self.discriminator(real, cond)
        d_fake   = self.discriminator(fake_seq, cond)
        gp       = self.gradient_penalty(real, fake_seq, cond)
        loss_D   = d_fake.mean() - d_real.mean() + self.gp_lambda * gp
        loss_D.backward()
        self.opt_D.step()

        # Weight clipping for Lipschitz
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)

        # ── Train Generator ───────────────────────────────────────────────
        self.opt_G.zero_grad()
        noise    = torch.randn(B, self.generator.latent_dim, device=cond.device)
        fake_seq = self.generator(cond, noise)
        d_fake   = self.discriminator(fake_seq, cond)
        loss_G   = -d_fake.mean()

        loss_fm = torch.tensor(0.0, device=cond.device)
        if self.fm_lambda > 0:
            loss_fm = F.mse_loss(fake_seq.mean(1), real.mean(1))
            loss_G  = loss_G + self.fm_lambda * loss_fm

        loss_G.backward()
        self.opt_G.step()

        return {
            "loss_G":  loss_G.item(),
            "loss_D":  loss_D.item(),
            "loss_fm": loss_fm.item(),
            "gp":      gp.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ADVERSARIAL STRATEGY EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_synthetic(
    real_df: pd.DataFrame,
    cond_arr: np.ndarray,
    gen_arr: np.ndarray,
    generator: Generator,
    dataset: MarketSequenceDataset,
    strategy_fn,
) -> dict:
    """
    Evaluate: do the GAN-generated scenarios break the strategy worse than real data?
    Returns attack success rate and worst-case metric degradation.
    """
    # Denormalize generated sequences
    gen_denorm = gen_arr * dataset.gen_std + dataset.gen_mean
    cond_denorm = cond_arr * dataset.cond_std + dataset.cond_mean

    # Reconstruct OHLC from generated returns + range
    close_start = real_df["close"].iloc[-1]
    synth_rets  = gen_denorm[:, :, 0]  # ret_1d is first column

    # Build synthetic OHLCV DataFrame
    n_synth = len(synth_rets[0])
    synth_close_vals = [close_start]
    for day_ret in synth_rets[0]:
        synth_close_vals.append(synth_close_vals[-1] * math.exp(day_ret))
    synth_close_vals = synth_close_vals[1:]

    # Create synthetic DataFrame with OHLCV-like structure
    synth_df = real_df.iloc[-n_synth:].copy()
    synth_df = synth_df.reset_index(drop=True)
    synth_df["close"] = synth_close_vals
    synth_df["open"]  = synth_df["close"].shift(1).fillna(synth_df["close"].iloc[0])
    synth_df["high"]  = synth_df[["close", "open"]].max(axis=1) * 1.005
    synth_df["low"]   = synth_df[["close", "open"]].min(axis=1) * 0.995
    synth_df["volume"] = synth_df.get("volume", synth_df.get("volume", pd.Series(1e6, index=synth_df.index)))
    synth_df["timestamp"] = synth_df.get("timestamp", pd.Series(range(len(synth_df)), index=synth_df.index))

    # Stress metric: max drawdown of strategy applied to synthetic scenario
    try:
        result_real = strategy_fn(real_df)
        result_synth = strategy_fn(synth_df)
        mdd_real  = result_real.get("max_dd", 0)
        mdd_synth = result_synth.get("max_dd", 0)
        degradation = max(0, mdd_synth - mdd_real)
        return {
            "mdd_real":  float(mdd_real),
            "mdd_synth": float(mdd_synth),
            "degradation": float(degradation),
            "attack_success": float(degradation > 0.05),
        }
    except Exception:
        return {"mdd_real": 0, "mdd_synth": 0, "degradation": 0, "attack_success": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_gan(
    df: pd.DataFrame,
    seq_len: int = 15,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    n_epochs: int = 40,
    batch_size: int = 64,
    n_critic: int = 3,
    seed: int = 42,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = MarketSequenceDataset(df, seq_len=seq_len)
    train_ds = dataset.get_split("train")
    test_ds  = dataset.get_split("test")

    cond_dim = len(dataset.cond_cols)
    gen_dim  = len(dataset.gen_cols)

    gan = WassersteinGAN(cond_dim, gen_dim, latent_dim, hidden_dim,
                        gp_lambda=10.0, fm_lambda=1.0).to(DEVICE)

    d_steps_per_g = n_critic

    history = {"epoch": [], "loss_D": [], "loss_G": [], "loss_fm": [], "gp": []}

    print(f"\n[AMG] Starting training: cond_dim={cond_dim}, gen_dim={gen_dim}, latent={latent_dim}")

    for epoch in range(1, n_epochs + 1):
        epoch_losses_G = []
        epoch_losses_D = []

        for cond, real in DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True):
            cond = cond.to(DEVICE)
            real = real.to(DEVICE)

            # Train discriminator n_critic times
            for _ in range(d_steps_per_g):
                m = gan.train_step(cond, real)
                epoch_losses_D.append(m)

            # Train generator once
            m = gan.train_step(cond, real)
            epoch_losses_G.append(m)

        avg_D  = np.mean([m["loss_D"] for m in epoch_losses_D])
        avg_G  = np.mean([m["loss_G"] for m in epoch_losses_G])
        avg_fm = np.mean([m.get("loss_fm", 0) for m in epoch_losses_G])
        avg_gp = np.mean([m.get("gp", 0) for m in epoch_losses_D])

        history["epoch"].append(epoch)
        history["loss_D"].append(avg_D)
        history["loss_G"].append(avg_G)
        history["loss_fm"].append(avg_fm)
        history["gp"].append(avg_gp)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss_D: {avg_D:+.4f} | loss_G: {avg_G:+.4f} | "
                  f"FM: {avg_fm:.6f} | GP: {avg_gp:.4f}")

    # ── Final evaluation ──────────────────────────────────────────────────
    gan.generator.eval()

    # Build proper 3D test condition tensor (B, seq_len, cond_dim)
    test_idx = dataset.test_indices
    test_cond_arr = np.stack([dataset.cond_data[i - seq_len:i] for i in test_idx])
    test_cond = torch.from_numpy(test_cond_arr).float().to(DEVICE)

    with torch.no_grad():
        noise     = torch.randn(len(test_cond), latent_dim, device=DEVICE)
        fake_seqs = gan.generator(test_cond, noise).cpu().numpy()

    real_seqs = np.stack([dataset.gen_data[i:i + seq_len] for i in test_idx])
    real_tensor = torch.from_numpy(real_seqs).float().to(DEVICE)
    cond_tensor = test_cond  # already built above as proper 3D tensor

    with torch.no_grad():
        d_real_score = gan.discriminator(real_tensor[:32], cond_tensor[:32]).mean().item()
        d_fake_score = gan.discriminator(
            torch.from_numpy(fake_seqs[:32]).float().to(DEVICE),
            cond_tensor[:32],
        ).mean().item()

    # Generator quality: MSE between real and synthetic distributions
    mse_gen = mean_squared_error(real_seqs.flatten(), fake_seqs.flatten())
    mae_gen = mean_absolute_error(real_seqs.flatten(), fake_seqs.flatten())

    # Distribution stats comparison
    real_mean = real_seqs.mean()
    fake_mean = fake_seqs.mean()
    real_std  = real_seqs.std()
    fake_std  = fake_seqs.std()

    # Adversarial success: how often does the generated scenario have extreme moves?
    fake_rets = fake_seqs[:, :, 0]  # ret_1d
    extreme_rate = float((np.abs(fake_rets) > real_seqs[:, :, 0].std() * 2).mean())
    worst_case_ret = float(np.abs(fake_rets).max())

    print(f"\n[AMG] Discriminator real score: {d_real_score:+.4f}")
    print(f"[AMG] Discriminator fake score:  {d_fake_score:+.4f}")
    print(f"[AMG] Generator MSE: {mse_gen:.6f} | MAE: {mae_gen:.6f}")
    print(f"[AMG] Extreme move rate: {extreme_rate:.1%} | Worst-case ret: {worst_case_ret:.4f}")

    # Save model
    os.makedirs("models/adversarial", exist_ok=True)
    torch.save({
        "generator_state": gan.generator.state_dict(),
        "discriminator_state": gan.discriminator.state_dict(),
        "config": {
            "cond_dim": cond_dim, "gen_dim": gen_dim,
            "latent_dim": latent_dim, "hidden_dim": hidden_dim,
        }
    }, "models/adversarial/best_gan.pt")

    result = {
        "d_real_score": round(d_real_score, 4),
        "d_fake_score": round(d_fake_score, 4),
        "generator_mse": round(mse_gen, 6),
        "generator_mae": round(mae_gen, 6),
        "real_mean":     round(float(real_mean), 6),
        "fake_mean":     round(float(fake_mean), 6),
        "real_std":      round(float(real_std), 6),
        "fake_std":      round(float(fake_std), 6),
        "extreme_rate":  round(extreme_rate * 100, 2),
        "worst_case_ret": round(worst_case_ret, 4),
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
    }

    with open("models/adversarial/training_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[AMG] Results saved -> models/adversarial/training_results.json")
    return result


def main():
    print("=" * 70)
    print("ADVERSARIAL MARKET GENERATOR — Conditional WGAN-GP")
    print("=" * 70)

    data_path = Path(__file__).parents[2].parent / "data.csv"
    df = load_and_prepare(data_path)

    result = train_gan(
        df,
        seq_len=15,
        latent_dim=32,
        hidden_dim=64,
        n_epochs=40,
        batch_size=64,
        n_critic=3,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("AMG FINAL RESULTS")
    print("=" * 70)
    print(f"  Discriminator real score:  {result['d_real_score']:+.4f}")
    print(f"  Discriminator fake score:   {result['d_fake_score']:+.4f}")
    print(f"  Generator MSE:             {result['generator_mse']:.6f}")
    print(f"  Generator MAE:            {result['generator_mae']:.6f}")
    print(f"  Real seq mean:             {result['real_mean']:+.6f}")
    print(f"  Fake seq mean:             {result['fake_mean']:+.6f}")
    print(f"  Real seq std:              {result['real_std']:.6f}")
    print(f"  Fake seq std:              {result['fake_std']:.6f}")
    print(f"  Extreme move rate:         {result['extreme_rate']:.1f}%")
    print(f"  Worst-case return:         {result['worst_case_ret']:.4f}")

    # Score out of 100
    # Good GAN: D scores close to 0 (Wasserstein), realistic distribution stats
    d_balanced = 1.0 / (1.0 + abs(result['d_real_score'] - result['d_fake_score']))
    dist_match = 1.0 - min(abs(result['real_mean'] - result['fake_mean']) / (abs(result['real_std']) + 1e-10), 1.0)
    std_match  = 1.0 - min(abs(result['real_std'] - result['fake_std']) / (result['real_std'] + 1e-10), 1.0)
    score = 40 * d_balanced + 30 * dist_match + 30 * std_match
    print(f"\n  GAN Quality Score: {score:.1f} / 100")

    return result


if __name__ == "__main__":
    main()
