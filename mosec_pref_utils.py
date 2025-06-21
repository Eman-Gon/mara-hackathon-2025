```
# -------------------------------------------------------------
# Algorithm 2 – line 2
#   wk ← RandomWeight(Ω)
# -------------------------------------------------------------
import random

def generate_random_preference_vector():
    """
    Generates a random (w_profit, w_efficiency, w_capacity) that sums to 1.
    ↳  Matches Algorithm 2, line 2  – picking a new random preference wk.
    """
    raw = [random.random() for _ in range(3)]
    total = sum(raw)
    return tuple(v / total for v in raw)


# -------------------------------------------------------------
# Algorithm 2 – lines 3  & 11
#   line 3 : E ← (wk, k)           (record current pref + episode)
#   line 11: wh ← EFscheme(E)      (return earliest vector)
# -------------------------------------------------------------
class EFMemory:
    """
    Earliest-First helper  
      • record(ep, vec)   → corresponds to Algorithm 2 line 3  
      • earliest()        → corresponds to Algorithm 2 line 11
    """
    def __init__(self):
        self.history = []            # [(episode, vector), …]

    def record(self, episode, vector):
        # Algorithm 2, line 3
        self.history.append((episode, vector))
        self.history.sort(key=lambda x: x[0])   # keep oldest first

    def earliest(self):
        # Algorithm 2, line 11
        return self.history[0][1] if self.history else None


# -------------------------------------------------------------
# Algorithm 2 – lines 13-16 (Q-network forward passes)
#   • Qkθ,i = Q-network(si , wk)
#   • Qhθ,i = Q-network(si , wh)
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreferenceQNet(nn.Module):
    """
    Preference-conditioned Q-network
    Used at Algorithm 2 lines 13-16 to compute Q(si, wk) or Q(si, wh)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + 3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, state, pref_vec):
        x = torch.cat([state, pref_vec], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)           # Q-values


# ----------------------------------------------------------------
# Tiny demo to show the mapping in action
# ----------------------------------------------------------------
if __name__ == "__main__":
    # --- Algorithm 2 line 2 ------------------------------------
    wk = generate_random_preference_vector()

    # episode counter (k)
    episode = 1

    # --- Algorithm 2 lines 3 & 11 ------------------------------
    ef = EFMemory()
    ef.record(episode, wk)            # line 3  (store current pref)
    wh = ef.earliest()                # line 11 (get earliest pref)

    # --- Algorithm 2 lines 13-16 demo --------------------------
    state_dim, action_dim = 10, 5
    net = PreferenceQNet(state_dim, action_dim)

    dummy_state = torch.randn(1, state_dim)
    wk_tensor  = torch.tensor([wk], dtype=torch.float32)
    wh_tensor  = torch.tensor([wh], dtype=torch.float32)

    q_wk = net(dummy_state, wk_tensor)   # Q(si, wk)  line 13
    q_wh = net(dummy_state, wh_tensor)   # Q(si, wh)  line 14

    print("Q-values with current pref wk :", q_wk)
    print("Q-values with earliest pref wh:", q_wh)
```
