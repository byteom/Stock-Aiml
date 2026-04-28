"""
PPO RL Agent for Strategy Parameter Optimization
==================================================

Actor-Critic PPO on a Gym-style trading environment.
State: TGNN embedding + stock features + portfolio state
Action: parameter adjustments (discrete buckets) or continuous position sizing
Reward: Sharpe - lambda_impact * Impact - lambda_cvar * CVaR

Components:
  1. TradingEnvironment: Gym-style env wrapping the backtester
  2. PPOMemory: experience replay buffer for PPO
  3. ActorCritic: shared-feature CNN + separate actor/critic heads
  4. PPOTrainer: clipping-based policy optimization
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PPO RL] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION (same as TGNN + GAN)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    """Load NIFTY 50, engineer features, return clean DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for c in df.columns:
        if c in ("date", "timestamp"):
            col_map[c] = "timestamp"
        elif c == "open":  col_map[c] = "open"
        elif c == "high":  col_map[c] = "high"
        elif c == "low":   col_map[c] = "low"
        elif c == "close": col_map[c] = "close"
    df = df.rename(columns=col_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %b %Y", dayfirst=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "volume" not in df.columns:
        df["volume"] = ((df["high"] - df["low"]) / df["high"].median() * 1e6).astype(float)

    df = df[df["high"] >= df["low"]].copy()
    df["close"] = df["close"].clip(df["low"] * 0.99, df["high"] * 1.01)

    for h in [1, 2, 5, 10, 20]:
        df[f"ret_{h}d"] = np.log(df["close"] / df["close"].shift(h)).fillna(0)

    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    ma20  = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - (ma20 - 2 * std20)) / (4 * std20 + 1e-10)

    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["log_ret"]  = df["ret_1d"]
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    df = df.fillna(0).reset_index(drop=True)
    print(f"[PPO Data] {len(df)} bars, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")
    return df


FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "rsi", "macd", "macd_sig", "bb_pos", "atr", "volume_ratio",
    "hl_range",
]


# ══════════════════════════════════════════════════════════════════════════════
#  TRADING ENVIRONMENT (Gym-style)
# ══════════════════════════════════════════════════════════════════════════════

class TradingEnvironment:
    """
    Gym-style trading environment wrapping a momentum strategy backtester.

    State:   [position_signals, portfolio_equity_norm, volatility_regime]
             — normalized feature vector of lookback window
    Action:  discrete buckets for (lookback_period, threshold, position_size)
             Total actions = Lb * Th * Ps action space combinations
    Reward:  risk-adjusted return (Sharpe - lambda * max_drawdown)

    Episode: one walk-forward window (e.g., 60 days)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = 60,
        initial_capital: float = 100_000.0,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.rng = np.random.default_rng(seed)

        # Walk-forward episode boundaries (70/15/15 train/val/test)
        n = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        self.train_range = (0,      train_end)
        self.val_range   = (train_end, val_end)
        self.test_range  = (val_end,   n)

        # Normalize using train period
        train_df = df.iloc[:train_end]
        self.feat_mean = train_df[feature_cols].mean().values
        self.feat_std  = train_df[feature_cols].std().values + 1e-8

        self._reset_env(self.train_range)

    def _reset_env(self, indices_range: tuple[int, int]):
        self.start_idx, self.end_idx = indices_range
        # Clamp end_idx to available data
        self.end_idx = min(self.end_idx, len(self.df))
        self.current_idx = self.start_idx + self.window_size
        self.current_idx = min(self.current_idx, self.end_idx - 1)
        self.done = False

        # Current parameter settings (evolved by RL)
        self.lookback    = 20    # days for signal lookback
        self.threshold  = 0.01  # entry threshold
        self.pos_size   = 1.0   # position size multiplier [0, 1]

        self.equity = [self.initial_capital]
        self.trades = []

    def reset(self, split: str = "train") -> np.ndarray:
        if split == "train":
            self._reset_env(self.train_range)
        elif split == "val":
            self._reset_env(self.val_range)
        else:
            self._reset_env(self.test_range)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build state vector: 20-bar lookback (ALWAYS FIXED) + equity ratio + pos_size."""
        i = self.current_idx
        feat_arr = self.df[self.feature_cols].values
        # ALWAYS use fixed 20-bar window — never self.lookback which changes per action
        window = feat_arr[max(0, i - 20):i]
        if len(window) < 20:
            pad = np.zeros((20 - len(window), window.shape[1]))
            window = np.vstack([pad, window])
        feat = (window - self.feat_mean) / self.feat_std
        equity_ratio = self.equity[-1] / self.initial_capital
        state = np.concatenate([
            feat.flatten(),  # 20 * 15 = 300
            [equity_ratio - 1.0],  # 1
            [self.pos_size],        # 1
        ])
        state = state.astype(np.float32)
        if state.shape[0] != self.state_dim:
            print(f"[DEBUG] state.shape={state.shape}, state_dim={self.state_dim}, "
                  f"len(window)={len(window)}, window.shape={window.shape}")
        assert state.shape[0] == self.state_dim, (
            f"State mismatch: got {state.shape[0]}, expected {self.state_dim}"
        )
        return state

    def _simulate_trade(self) -> dict:
        """Run one step of momentum strategy simulation."""
        i = self.current_idx
        lookback = min(self.lookback, i - self.window_size)

        # Momentum signal
        ret_lookback = float(self.df["ret_1d"].iloc[i - lookback:i].sum())
        vol          = float(self.df["vol_5d"].iloc[i - 1]) if i > 0 else 0.01

        if ret_lookback > self.threshold * lookback:
            direction = 1   # long
        elif ret_lookback < -self.threshold * lookback:
            direction = -1  # short
        else:
            direction = 0   # cash

        ret_today = float(self.df["ret_1d"].iloc[i])
        pos_value = self.equity[-1] * self.pos_size
        pnl = pos_value * direction * ret_today

        # Slippage cost
        slippage = pos_value * vol * 0.001

        self.equity.append(max(1.0, self.equity[-1] + pnl - slippage))

        trade = {
            "idx": i,
            "direction": direction,
            "pnl": pnl,
            "slippage": slippage,
            "equity": self.equity[-1],
        }
        self.trades.append(trade)
        return trade

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Discrete action: 0=conservative(10d,0.5%), 1=momentum(20d,1%), 2=aggressive(20d,2%), 3=high-conv(30d,1%)
        """
        params = [
            (10, 0.005, 0.5),
            (20, 0.01,  1.0),
            (20, 0.02,  1.0),
            (30, 0.01,  1.0),
        ]
        self.lookback, self.threshold, self.pos_size = params[action % len(params)]

        trade = self._simulate_trade()
        self.current_idx += 1

        next_state = self._get_state()

        # Reward: equity growth rate
        prev_equity = self.equity[-2] if len(self.equity) > 1 else self.initial_capital
        reward = (self.equity[-1] - prev_equity) / prev_equity

        self.done = (self.current_idx >= self.end_idx - 1)

        info = {
            "equity": self.equity[-1],
            "direction": trade["direction"],
        }
        return next_state, reward, self.done, info

    @property
    def state_dim(self) -> int:
        return 20 * len(self.feature_cols) + 2  # always 20-bar lookback

    @property
    def num_actions(self) -> int:
        return 4  # 4 discrete strategy parameter choices


# ══════════════════════════════════════════════════════════════════════════════
#  PPO MEMORY (Experience Replay Buffer)
# ══════════════════════════════════════════════════════════════════════════════

class PPOMemory:
    """Stores trajectories for PPO policy gradient updates."""

    def __init__(self):
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.log_probs  = []
        self.values     = []
        self.dones      = []

    def store(self, state: np.ndarray, action: int, reward: float,
              log_prob: float, value: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════════════════
#  ACTOR-CRITIC NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Shared-feature + separate Actor/Critic heads.
    Actor:  outputs action probabilities (logits -> policy)
    Critic: outputs state value estimate (V(s))
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: policy over actions
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Critic head: state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(state)
        logits    = self.actor(shared_out)
        value     = self.critic(shared_out)
        return logits, value

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy. Returns: action, log_prob, value."""
        logits, value = self.forward(state)
        probs   = F.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """
    Proximal Policy Optimization trainer.

    Algorithm:
      1. Collect trajectories using current policy
      2. Compute GAE (Generalized Advantage Estimation)
      3. Update policy via clipped PPO objective (multiple epochs)
      4. Value function baseline update
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        hidden_dim: int = 128,
    ):
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.k_epochs     = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef

        self.policy = ActorCritic(state_dim, num_actions, hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)

        self.memory = PPOMemory()

    def compute_gae(self, rewards: list, values: list, dones: list) -> tuple[list, list]:
        """Compute Generalized Advantage Estimation."""
        advantages  = []
        returns     = []
        running_adv = 0.0
        running_ret = 0.0

        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            running_adv = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * running_adv
            running_ret = rewards[t] + self.gamma * (1 - dones[t]) * running_ret
            advantages.insert(0, running_adv)
            returns.insert(0, running_ret)
            next_value = values[t]

        # Normalize advantages
        if advantages:
            adv_np = np.array(advantages, dtype=np.float32)
            adv_mean = adv_np.mean()
            adv_std  = adv_np.std() + 1e-8
            advantages = [float((a - adv_mean) / adv_std) for a in advantages]

        return advantages, returns

    def update(self) -> dict:
        """Perform PPO policy update from collected memory."""
        if len(self.memory.states) == 0:
            return {"loss_total": 0.0}

        states    = torch.from_numpy(np.array(self.memory.states)).float().to(DEVICE)
        actions   = torch.tensor(self.memory.actions, dtype=torch.long).to(DEVICE)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(DEVICE)
        rewards   = self.memory.rewards
        values    = self.memory.values
        dones     = self.memory.dones

        advantages, returns = self.compute_gae(rewards, values, dones)
        returns_tensor     = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        # Multiple epochs of PPO update
        total_loss = 0.0
        for _ in range(self.k_epochs):
            logits, values_pred = self.policy.forward(states)
            values_pred = values_pred.squeeze(-1)

            probs     = F.softmax(logits, dim=-1)
            dist      = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy   = dist.entropy().mean()

            # Ratio for PPO clipping
            ratio = torch.exp(log_probs - old_log_probs.detach())

            surr1 = ratio * torch.tensor(advantages, device=DEVICE)
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surr2 = surr2 * torch.tensor(advantages, device=DEVICE)

            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = F.mse_loss(values_pred, returns_tensor)
            entropy_loss = -self.entropy_coef * entropy

            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss = loss.item()

        self.memory.clear()
        return {
            "loss_total":   round(total_loss, 6),
            "policy_loss":   round(policy_loss.item(), 6),
            "value_loss":   round(value_loss.item(), 6),
            "entropy":       round(entropy.item(), 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FULL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo(
    df: pd.DataFrame,
    num_episodes: int = 300,
    max_steps_per_episode: int = 120,
    seed: int = 42,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = TradingEnvironment(df, FEATURE_COLS, window_size=60, seed=seed)

    trainer = PPOTrainer(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        hidden_dim=128,
    )

    history = {"episode": [], "train_reward": [], "val_reward": [],
               "equity_train": [], "equity_val": [], "loss": []}

    best_val = -9999.0
    best_policy_state = None

    print(f"\n[PPO] state_dim={env.state_dim}, num_actions={env.num_actions}")

    for ep in range(1, num_episodes + 1):
        # ── Collect train episode ─────────────────────────────────────────
        state = env.reset(split="train")
        ep_reward = 0.0

        for step in range(max_steps_per_episode):
            state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, log_prob, value = trainer.policy.get_action(state_t)

            next_state, reward, done, info = env.step(action.item())
            trainer.memory.store(state, action.item(), reward, log_prob.item(),
                                 value.item(), done)

            state = next_state
            ep_reward += reward

            if done:
                break

        # PPO update after each episode
        loss_dict = trainer.update()

        train_equity = env.equity[-1]
        train_reward = ep_reward

        # ── Validation episode ─────────────────────────────────────────────
        val_state = env.reset(split="val")
        ep_reward_val = 0.0
        for step in range(max_steps_per_episode):
            state_t = torch.from_numpy(val_state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _ = trainer.policy.get_action(state_t)
            val_state, reward, done, _ = env.step(action.item())
            ep_reward_val += reward
            if done:
                break

        val_equity = env.equity[-1]

        history["episode"].append(ep)
        history["train_reward"].append(round(train_reward, 4))
        history["val_reward"].append(round(ep_reward_val, 4))
        history["equity_train"].append(round(train_equity, 2))
        history["equity_val"].append(round(val_equity, 2))
        history["loss"].append(loss_dict.get("loss_total", 0.0))

        # Track best validation
        if ep_reward_val > best_val:
            best_val = ep_reward_val
            best_policy_state = {k: v.clone() for k, v in trainer.policy.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | TrainR: {train_reward:+.4f} | ValR: {ep_reward_val:+.4f} | "
                  f"TrainEQ: {train_equity:,.0f} | ValEQ: {val_equity:,.0f} | Loss: {loss_dict.get('loss_total', 0):.4f}")

    # ── Restore best policy ─────────────────────────────────────────────────
    if best_policy_state is not None:
        trainer.policy.load_state_dict(best_policy_state)

    # ── Final test evaluation ──────────────────────────────────────────────
    test_state = env.reset(split="test")
    ep_reward_test = 0.0
    for step in range(max_steps_per_episode):
        state_t = torch.from_numpy(test_state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, _ = trainer.policy.get_action(state_t)
        test_state, reward, done, _ = env.step(action.item())
        ep_reward_test += reward
        if done:
            break

    test_equity = env.equity[-1]
    print(f"\n[PPO] Test equity: {test_equity:,.2f} | Test reward: {ep_reward_test:+.4f}")

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs("stock-aiml/models/rl", exist_ok=True)
    torch.save({
        "policy_state": trainer.policy.state_dict(),
        "config": {
            "state_dim": env.state_dim,
            "num_actions": env.num_actions,
            "hidden_dim": 128,
        }
    }, "stock-aiml/models/rl/best_ppo.pt")

    result = {
        "test_equity":     round(test_equity, 2),
        "test_reward":     round(ep_reward_test, 4),
        "best_val_reward": round(best_val, 4),
        "final_train_eq":  round(history["equity_train"][-1], 2),
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
    }

    with open("stock-aiml/models/rl/training_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PPO RL TRADING AGENT — Strategy Parameter Optimization")
    print("=" * 70)

    data_path = Path("C:/Users/anwee/Desktop_1/Learning-Season/Stock-Aiml/data.csv")
    df = load_and_prepare(data_path)

    result = train_ppo(
        df,
        num_episodes=200,
        max_steps_per_episode=60,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("PPO RL FINAL RESULTS")
    print("=" * 70)
    print(f"  Final train equity:  {result['final_train_eq']:,.2f}")
    print(f"  Best val reward:       {result['best_val_reward']:+.4f}")
    print(f"  Test equity:           {result['test_equity']:,.2f}")
    print(f"  Test reward:           {result['test_reward']:+.4f}")

    # Score out of 100: equity growth + risk-adjusted reward
    eq_score = min(result['test_equity'] / 100_000.0, 2.0) * 50  # up to 50 pts for equity growth
    rw_score = min(max(result['test_reward'] * 10 + 0.5, 0.0), 1.0) * 50  # up to 50 pts for reward
    score = eq_score + rw_score
    print(f"\n  RL Agent Score: {score:.1f} / 100")

    return result


if __name__ == "__main__":
    main()
