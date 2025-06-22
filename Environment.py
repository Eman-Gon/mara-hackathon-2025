class Environment:
    """Enhanced Environment based on research implementation"""

    P_TOTAL = 1_000_000  # total available watts (1MW)

    def __init__(self):
        self.inventory = INVENTORY
        self.site = sample_site.copy()  # Available machines on site
        self.time_step = 0

    def get_simulated_prices(self):
        """Simulate fluctuating market prices"""
        # Simulate price fluctuations based on sample_state
        base_time = time.time() + self.time_step * 300  # 5-minute intervals

        # Start from research sample prices and add fluctuations
        energy_price = sample_state["energy_price"] + 0.1 * np.sin(base_time / 1000) + 0.05 * np.random.randn()
        hash_price = sample_state["hash_price"] + 0.3 * np.cos(base_time / 1500) + 0.1 * np.random.randn()
        token_price = sample_state["token_price"] + 0.1 * np.sin(base_time / 2000) + 0.05 * np.random.randn()

        return {
            "energy_price": max(0.5, energy_price),
            "hash_price": max(2.0, hash_price),
            "token_price": max(0.4, token_price)
        }

    def get_current_state(self):
        """Get current state representation"""
        prices = self.get_simulated_prices()

        # State: [energy_price, hash_price, token_price, power_usage_ratio, mining_efficiency, token_efficiency]
        current_power_usage = self.site.get("total_power_used", 0)
        power_ratio = current_power_usage / self.P_TOTAL

        # Calculate efficiency metrics
        mining_efficiency = self.site.get("revenue", {}).get("immersion_miners", 0) / max(1, self.site.get("power", {}).get("immersion_miners", 1))
        token_efficiency = self.site.get("revenue", {}).get("asic_compute", 0) / max(1, self.site.get("power", {}).get("asic_compute", 1))

        state = np.array([
            prices["energy_price"],
            prices["hash_price"],
            prices["token_price"],
            power_ratio,
            mining_efficiency / 1000,  # Normalize
            token_efficiency / 1000    # Normalize
        ])

        return state

    def step(self, action, pref_vector):
        """Execute action using greedy allocation strategy from research"""
        self.time_step += 1

        # Get action ratios
        if action >= len(ACTION_SPACE_RATIOS):
            action = 0
        infer_ratio, mining_ratio = ACTION_SPACE_RATIOS[action]

        # Get current prices
        prices = self.get_simulated_prices()

        # Calculate power allocations
        P_infer_rem = infer_ratio * self.P_TOTAL
        P_mine_rem = mining_ratio * self.P_TOTAL

        # Check if allocation exceeds total power constraint
        if (P_infer_rem + P_mine_rem) > self.P_TOTAL:
            # Invalid action - penalize heavily
            reward = -10000
            return self.get_current_state(), reward

        # ==========================================
        # GREEDY INFERENCE ALLOCATION (from research)
        # ==========================================

        # Sort inference machines by tokens-per-watt descending
        inf_order = sorted(
            self.inventory['inference'].items(),
            key=lambda kv: kv[1]['tokens'] / kv[1]['power'],
            reverse=True
        )

        power_inf_used = 0
        tokens_generated = 0

        for inf_name, specs in inf_order:
            unit_p = specs['power']
            unit_t = specs['tokens']
            # How many you actually have on-site
            available = self.site.get(f"{inf_name}_compute", 0)
            # How many you *could* power
            max_by_power = int(P_infer_rem // unit_p) if unit_p > 0 else 0
            n = min(available, max_by_power)
            if n <= 0:
                continue
            # Consume power & produce tokens
            used = n * unit_p
            power_inf_used += used
            tokens_generated += n * unit_t
            P_infer_rem -= used
            # Stop if no more inference power budget
            if P_infer_rem < min([m['power'] for m in self.inventory['inference'].values()]):
                break

        # ==========================================
        # GREEDY MINING ALLOCATION (from research)
        # ==========================================

        # Sort miners by (hashrate-per-watt, hashrate) descending
        mine_order = sorted(
            self.inventory['miners'].items(),
            key=lambda kv: (kv[1]['hashrate'] / kv[1]['power'], kv[1]['hashrate']),
            reverse=True
        )

        power_mine_used = 0
        hash_generated = 0

        for mine_name, specs in mine_order:
            unit_p = specs['power']
            unit_h = specs['hashrate']
            available = self.site.get(f"{mine_name}_miners", 0)
            max_by_power = int(P_mine_rem // unit_p) if unit_p > 0 else 0
            n = min(available, max_by_power)
            if n <= 0:
                continue
            used = n * unit_p
            power_mine_used += used
            hash_generated += n * unit_h
            P_mine_rem -= used
            if P_mine_rem < min([m['power'] for m in self.inventory['miners'].values()]):
                break

        # ==========================================
        # PRICING & REVENUE CALCULATION (from research)
        # ==========================================

        ep = prices['energy_price']
        hp = prices['hash_price']
        tp = prices['token_price']

        mining_revenue = hash_generated * hp
        infer_revenue = tokens_generated * tp
        total_revenue = mining_revenue + infer_revenue

        # COSTS
        total_cost = self.P_TOTAL * ep  # Pay for full power capacity

        # METRICS
        profit = total_revenue - total_cost
        energy_usage = power_inf_used + power_mine_used
        compute_idle = self.P_TOTAL - energy_usage

        # ========================================
        # MULTI-OBJECTIVE REWARD (FIXED VERSION)
        # ========================================

        # Fix: Use array indexing instead of dot notation
        reward = (
            profit * pref_vector[0] -              # profit preference
            energy_usage * pref_vector[1] * 0.001 - # energy saving preference
            compute_idle * pref_vector[2] * 0.0001   # compute usage preference
        )

        # Update site state for next iteration
        self.site.update({
            "total_power_used": int(energy_usage),
            "total_power_cost": total_cost,
            "total_revenue": total_revenue,
            "power": {
                "inference_used": int(power_inf_used),
                "mining_used": int(power_mine_used)
            },
            "revenue": {
                "mining_revenue": mining_revenue,
                "inference_revenue": infer_revenue
            }
        })

        print(f" Power: Inference={power_inf_used:.0f}W, Mining={power_mine_used:.0f}W, Total={energy_usage:.0f}W")
        print(f" Revenue: Mining=${mining_revenue:.2f}, Inference=${infer_revenue:.2f}, Cost=${total_cost:.2f}")
        print(f" Reward: {reward:.2f} (Profit={profit:.2f}, Usage={energy_usage:.0f}, Idle={compute_idle:.0f})")

        # Generate next state
        new_state = self.get_current_state()

        return new_state, reward
