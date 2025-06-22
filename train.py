
def Learning_Loop_with_Adaptation(ep_num, site_num, sample_coef=1.0):
    """Enhanced learning loop with adaptation error tracking"""

    # Initialize components
    agent = Agent()
    environment = Environment()
    ef_memory = EFMemory()
    exp_replay = ExperienceReplayBuffer(capacity=10000)
    adaptation_tracker = AdaptationErrorTracker(window_size=10)

    gamma = 0.99
    batch_size = 32
    epsilon = 0.1
    epsilon_decay = 0.995
    epsilon_min = 0.01

    episode_rewards = []
    adaptation_errors = []
    previous_pref = None

    print(" Starting Multi-Objective RL Training with Adaptation Error...")

    for ep in range(ep_num):
        print(f"\n Episode {ep+1}/{ep_num}")

        # Generate preference vector for this episode
        select_pref_vect = generate_random_preference_vector()
        ef_memory.record(ep, select_pref_vect)

        # Calculate adaptation error if preference changed
        adaptation_error = 0.0
        if previous_pref is not None:
            adaptation_error = calculate_adaptation_error(
                agent, environment, select_pref_vect, previous_pref, adaptation_tracker
            )
            adaptation_errors.append(adaptation_error)

        episode_reward = 0

        for site in range(site_num):
            # Get current state
            cur_state = environment.get_current_state()

            # Get Q-values and select action
            Q_vals = agent.QNet(cur_state, select_pref_vect)
            action = epsilon_greedy(Q_vals, epsilon)

            # Execute action (now using ratio-based allocation)
            new_state, base_reward = environment.step(action, select_pref_vect)

            # Apply adaptation error penalty
            adaptation_penalty = adaptation_tracker.get_adaptation_penalty()
            final_reward = base_reward - adaptation_penalty

            episode_reward += final_reward

            # Track adaptation
            adaptation_tracker.add_experience(select_pref_vect, final_reward)

            # Store experience
            exp_replay.add(cur_state, select_pref_vect, action, new_state, final_reward)

            # Training step if we have enough experiences
            if exp_replay.size() >= batch_size:
                # Sample experiences using NER
                trans = Sample_NER(sample_coef, batch_size, exp_replay, select_pref_vect, cur_state)

                if len(trans) > 0:
                    # Get earliest preference for dual learning
                    ef_pref_vect = ef_memory.earliest()

                    losses = []

                    for exp_data, similarity in trans:
                        exp = exp_data

                        # Convert to tensors
                        s_i = torch.FloatTensor(exp.state).unsqueeze(0).to(device)
                        a_i = exp.action
                        r_i = torch.FloatTensor([exp.reward]).to(device)
                        s_next_i = torch.FloatTensor(exp.new_state).unsqueeze(0).to(device)

                        # Q-values for current preferences
                        Q_select_pref = agent.QNet(s_i, select_pref_vect)
                        Q_ef_pref = agent.QNet(s_i, ef_pref_vect)

                        Q_select_pref_i = Q_select_pref[0, a_i]
                        Q_ef_pref_i = Q_ef_pref[0, a_i]

                        # Target Q-values
                        with torch.no_grad():
                            target_Q_select = agent.target_QNet(s_next_i, select_pref_vect)
                            target_Q_ef = agent.target_QNet(s_next_i, ef_pref_vect)

                            target_Q_select_max = target_Q_select.max(dim=-1)[0]
                            target_Q_ef_max = target_Q_ef.max(dim=-1)[0]

                            y_select_pref_i = r_i + gamma * target_Q_select_max
                            y_ef_pref_i = r_i + gamma * target_Q_ef_max

                        # Loss for this transition
                        loss_i = 0.5 * (
                            F.mse_loss(Q_select_pref_i, y_select_pref_i) +
                            F.mse_loss(Q_ef_pref_i, y_ef_pref_i)
                        )
                        losses.append(loss_i)

                    if losses:
                        # Backprop update
                        total_loss = torch.stack(losses).mean()
                        agent.optimizer.zero_grad()
                        total_loss.backward()
                        agent.optimizer.step()

        # Update target network periodically
        if ep % 10 == 0:
            agent.target_QNet.load_state_dict(agent.QNet.state_dict())
            print(f" Target network updated at episode {ep}")

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)

        # Update previous preference
        previous_pref = select_pref_vect.copy()

        print(f" Episode Reward: {episode_reward:.2f}")
        print(f" Preference Vector: Profit={select_pref_vect[0]:.2f}, Energy={select_pref_vect[1]:.2f}, Compute={select_pref_vect[2]:.2f}")
        print(f" Adaptation Error: {adaptation_error:.3f}")
        print(f" Epsilon: {epsilon:.3f}")

        # Show progress every 20 episodes
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_adaptation_error = np.mean(adaptation_errors[-20:]) if adaptation_errors else 0
            print(f" Average Reward (last 20): {avg_reward:.2f}")
            print(f" Average Adaptation Error (last 20): {avg_adaptation_error:.3f}")

    return episode_rewards, agent, environment, adaptation_errors

# ========================
# SECTION 10: ADAPTATION ERROR MODULE - COMPLETE IMPLEMENTATION
# ========================

class AdaptationErrorTracker:
    """
    Tracks adaptation error when preferences change
    Measures how well the agent adapts to new preference vectors
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.preference_history = []
        self.reward_history = []
        self.adaptation_errors = []

    def add_experience(self, preference_vector, reward):
        """Add new preference-reward pair"""
        self.preference_history.append(preference_vector.copy())
        self.reward_history.append(reward)

        # Keep only recent history
        if len(self.preference_history) > self.window_size * 2:
            self.preference_history = self.preference_history[-self.window_size * 2:]
            self.reward_history = self.reward_history[-self.window_size * 2:]

    def compute_adaptation_error(self, current_preference, expected_reward, actual_reward):
        """
        Compute adaptation error when preference changes
        """
        if len(self.preference_history) < self.window_size:
            return 0.0

        # Find similar past preferences
        similar_rewards = []
        for i, past_pref in enumerate(self.preference_history[-self.window_size:]):
            similarity = 1.0 - np.linalg.norm(current_preference - past_pref)
            if similarity > 0.8:  # Similar preference threshold
                similar_rewards.append(self.reward_history[-(self.window_size-i)])

        if len(similar_rewards) > 0:
            expected_performance = np.mean(similar_rewards)
            adaptation_error = abs(actual_reward - expected_performance)
            self.adaptation_errors.append(adaptation_error)
            return adaptation_error

        return 0.0

    def get_adaptation_penalty(self):
        """Get penalty based on recent adaptation errors"""
        if len(self.adaptation_errors) == 0:
            return 0.0

        recent_errors = self.adaptation_errors[-5:]  # Last 5 errors
        avg_error = np.mean(recent_errors)

        # Penalty increases with adaptation error
        penalty = min(avg_error * 0.1, 100.0)  # Cap at 100
        return penalty

def calculate_adaptation_error(agent, environment, current_pref, previous_pref=None, adaptation_tracker=None):
    """
    Calculate adaptation error when switching between preferences
    Based on research implementation using Q-learning approach
    """
    if previous_pref is None or adaptation_tracker is None:
        return 0.0

    # Test current performance with current preference
    test_state = environment.get_current_state()

    with torch.no_grad():
        q_values_current = agent.QNet(test_state, current_pref)
        q_values_previous = agent.QNet(test_state, previous_pref)

    # Calculate preference distance
    pref_distance = np.linalg.norm(current_pref - previous_pref)

    # Calculate Q-value difference
    q_diff = torch.mean(torch.abs(q_values_current - q_values_previous)).item()

    # Adaptation error: high when preferences change a lot but Q-values don't adapt proportionally
    adaptation_error = pref_distance * 10.0 - q_diff
    adaptation_error = max(0.0, adaptation_error)  # Only positive errors

    return adaptation_error

# ========================
# Q-LEARNING ADAPTATION ERROR BENCHMARK (From Research)
# ========================

def calculate_q_learning_adaptation_error():
    """
    Research-based Q-learning adaptation error calculation
    This implements the exact method from your research code
    """

    print(" Running Q-Learning Adaptation Error Benchmark...")

    # Discretization (from research)
    energy_bins = np.linspace(0.5, 1.0, 5)
    hash_bins = np.linspace(7, 12, 5)
    token_bins = np.linspace(1.5, 3.5, 5)

    # Actions (from research)
    actions = [0.0, 0.25, 0.5, 0.75, 1.0]  # mining ratios
    Q = np.zeros((6, 6, 6, len(actions)))  # Q-table

    # Reward function (from research)
    def get_reward(mining_ratio, energy_price, hash_price, token_price, preference):
        P_total = 1_000_000
        P_mining = mining_ratio * P_total
        P_infer = (1 - mining_ratio) * P_total
        hash_per_watt = 1.0
        token_per_watt = 3.33

        mining_revenue = P_mining * hash_price * hash_per_watt
        infer_revenue = P_infer * token_price * token_per_watt
        total_revenue = mining_revenue + infer_revenue

        cost = P_total * energy_price
        profit = total_revenue - cost
        energy_eff = total_revenue / P_total
        compute_score = infer_revenue

        return (
            preference[0] * profit +
            preference[1] * energy_eff +
            preference[2] * compute_score
        )

    # Training parameters (from research)
    alpha, gamma, epsilon = 0.4, 0.3, 0.8
    episodes = 1000  # Reduced for faster execution
    pref_train = [0.6, 0.2, 0.2]  # Train preference: Profit-weighted
    train_rewards = []

    # Training phase
    for ep in range(episodes):
        e = np.random.uniform(0.5, 1.0)
        h = np.random.uniform(7.0, 12.0)
        t = np.random.uniform(1.5, 3.5)

        s = (np.digitize(e, energy_bins), np.digitize(h, hash_bins), np.digitize(t, token_bins))
        a = np.random.randint(len(actions)) if random.random() < epsilon else np.argmax(Q[s])

        mining_ratio = actions[a]
        r = get_reward(mining_ratio, e, h, t, pref_train)
        train_rewards.append(r)

        # Next state
        e2, h2, t2 = np.random.uniform(0.5, 1.0), np.random.uniform(7.0, 12.0), np.random.uniform(1.5, 3.5)
        s2 = (np.digitize(e2, energy_bins), np.digitize(h2, hash_bins), np.digitize(t2, token_bins))
        a2 = np.argmax(Q[s2])

        Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])

    # Testing phase with new preference
    pref_test = [0.3, 0.1, 0.6]  # New preference (compute-focused)
    test_rewards = []

    for _ in range(200):  # Reduced for faster execution
        e = np.random.uniform(0.5, 1.0)
        h = np.random.uniform(7.0, 12.0)
        t = np.random.uniform(1.5, 3.5)

        s = (np.digitize(e, energy_bins), np.digitize(h, hash_bins), np.digitize(t, token_bins))
        a = np.argmax(Q[s])

        mining_ratio = actions[a]
        r = get_reward(mining_ratio, e, h, t, pref_test)
        test_rewards.append(r)

    # Adaptation error calculation (from research)
    R_optimal = sum(train_rewards[-200:])  # Last 200 training rewards
    R_test = sum(test_rewards)
    adaptation_error = abs(R_test - R_optimal) / abs(R_optimal) if R_optimal != 0 else 0

    print(f" Q-Learning Benchmark Results:")
    print(f"   Optimal reward (train): {R_optimal:.2f}")
    print(f"   Actual reward (test): {R_test:.2f}")
    print(f"   Adaptation Error: {adaptation_error:.4f}")

    return adaptation_error, train_rewards, test_rewards

# ========================
# SECTION 11: ENHANCED LEARNING LOOP WITH ADAPTATION ERROR
# ========================

def test_reward_function():
    """ Test and demonstrate the reward function with research-based environment"""

    print(" TESTING REWARD FUNCTION (Research Version)")
    print("=" * 60)

    env = Environment()

    # Test different preference vectors
    test_preferences = [
        np.array([0.8, 0.1, 0.1]),    # Profit-focused
        np.array([0.3, 0.6, 0.1]),    # Energy-saving focused
        np.array([0.3, 0.1, 0.6]),    # Compute-utilization focused
        np.array([0.33, 0.33, 0.34])  # Balanced
    ]

    test_actions = [0, 3, 5, 7, 10]  # Different allocation ratio strategies

    for i, pref in enumerate(test_preferences):
        print(f"\n Preference {i+1}: Profit={pref[0]:.1f}, Energy={pref[1]:.1f}, Compute={pref[2]:.1f}")

        for action in test_actions:
            if action < len(ACTION_SPACE_RATIOS):
                infer_ratio, mining_ratio = ACTION_SPACE_RATIOS[action]
                new_state, reward = env.step(action, pref)

                print(f"   Action {action}: Inference={infer_ratio:.1f}, Mining={mining_ratio:.1f} -> Reward: {reward:.2f}")
            else:
                break

    return env

# Add this to main function call
# ========================
# SECTION 12: MAIN EXECUTION WITH ADAPTATION ERROR
# ========================

def main():
    """Main execution function with adaptation error tracking"""

    print(" MOMI MARA - Multi-Objective Trading System")
    print(" Running in Simulation Mode (No API Required)")
    print(" WITH ADAPTATION ERROR TRACKING")
    print("=" * 60)

    # Run Q-learning benchmark first
    print("\n Running Q-Learning Adaptation Error Benchmark...")
    q_adaptation_error, q_train_rewards, q_test_rewards = calculate_q_learning_adaptation_error()

    # Test reward function
    print("\n Testing reward function...")
    test_reward_function()

    # Run training with adaptation error
    print("\n🏋️ Starting Multi-Objective RL training with adaptation error...")
    rewards, trained_agent, env, adaptation_errors = Learning_Loop_with_Adaptation(
        ep_num=100,
        site_num=5,
        sample_coef=1.0
    )

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot RL training rewards
    ax1.plot(rewards)
    ax1.set_title('Multi-Objective RL Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    # Plot RL adaptation errors
    if adaptation_errors:
        ax2.plot(adaptation_errors, color='red')
        ax2.set_title('RL Adaptation Errors Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Adaptation Error')
        ax2.grid(True)

    # Plot Q-learning benchmark
    ax3.plot(q_train_rewards, label="Q-Learning Training", alpha=0.7)
    ax3.axhline(y=np.mean(q_test_rewards), color='red', linestyle='--',
                label=f"Q-Learning Test (Adaptation Error: {q_adaptation_error:.3f})")
    ax3.set_title('Q-Learning Benchmark')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True)

    # Comparison plot
    if adaptation_errors:
        rl_adaptation_error = np.mean(adaptation_errors[-20:]) if len(adaptation_errors) >= 20 else np.mean(adaptation_errors)
        methods = ['Q-Learning\nBenchmark', 'Multi-Objective\nRL (Ours)']
        errors = [q_adaptation_error, rl_adaptation_error]

        ax4.bar(methods, errors, color=['orange', 'blue'], alpha=0.7)
        ax4.set_title('Adaptation Error Comparison')
        ax4.set_ylabel('Adaptation Error')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(errors):
            ax4.text(i, v + max(errors) * 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print(" Training completed!")

    # Analyze adaptation performance
    if adaptation_errors:
        avg_adaptation_error = np.mean(adaptation_errors)
        final_adaptation_errors = np.mean(adaptation_errors[-20:]) if len(adaptation_errors) >= 20 else np.mean(adaptation_errors)
        improvement = ((avg_adaptation_error - final_adaptation_errors) / avg_adaptation_error * 100) if avg_adaptation_error != 0 else 0

        print(f"\n ADAPTATION ANALYSIS:")
        print(f"   Q-Learning Benchmark Adaptation Error: {q_adaptation_error:.3f}")
        print(f"   Our Multi-Objective RL Average: {avg_adaptation_error:.3f}")
        print(f"   Our Multi-Objective RL Final: {final_adaptation_errors:.3f}")
        print(f"   Our Method Improvement: {improvement:.1f}%")

        if final_adaptation_errors < q_adaptation_error:
            print(f"   Our method performs {((q_adaptation_error - final_adaptation_errors) / q_adaptation_error * 100):.1f}% better!")
        else:
            print(f"    Q-Learning benchmark performs {((final_adaptation_errors - q_adaptation_error) / q_adaptation_error * 100):.1f}% better")

    # Test the trained agent
    print("\n Testing trained agent with different preferences...")
    test_preferences = [
        np.array([0.8, 0.1, 0.1]),  # Profit-focused
        np.array([0.3, 0.6, 0.1]),  # Energy-saving focused
        np.array([0.3, 0.1, 0.6]),  # Compute-utilization focused
    ]

    for i, test_pref in enumerate(test_preferences):
        test_state = env.get_current_state()

        with torch.no_grad():
            q_values = trained_agent.QNet(test_state, test_pref)
            best_action = q_values.argmax().item()

        pref_type = ["Profit-focused", "Energy-focused", "Compute-focused"][i]
        infer_ratio, mining_ratio = ACTION_SPACE_RATIOS[best_action]

        print(f" {pref_type}: Preference {test_pref} -> Action {best_action}")
        print(f"   Allocation Ratios: {infer_ratio:.1f} inference, {mining_ratio:.1f} mining")
        print(f"   Power Split: {infer_ratio*100:.0f}% inference, {mining_ratio*100:.0f}% mining")

    return trained_agent, env, rewards, adaptation_errors

if __name__ == "__main__":
    print(" STARTING MOMI MARA MULTI-OBJECTIVE TRADING SYSTEM")
    print(" With Research-Based Adaptation Error Analysis")
    print("=" * 70)

    try:
        trained_agent, environment, reward_history, adaptation_error_history = main()

        print("\n SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        print(f" Total episodes trained: {len(reward_history)}")
        print(f" Final average reward: {np.mean(reward_history[-10:]):.2f}")
        if adaptation_error_history:
            print(f" Final adaptation error: {np.mean(adaptation_error_history[-10:]):.3f}")

    except Exception as e:
        print(f"\n Error during execution: {e}")
        print(" Please check all dependencies are installed:")
        print("   pip install torch numpy matplotlib pandas requests")
        import traceback
        traceback.print_exc()