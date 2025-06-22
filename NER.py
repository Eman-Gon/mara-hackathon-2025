def compute_similarity(cur_state, sampled_state, cur_pref, sampled_pref):
    state_similarity = np.linalg.norm(cur_state - sampled_state, ord=2)
    preference_similarity = np.linalg.norm(cur_pref - sampled_pref, ord=2)
    return state_similarity, preference_similarity

def Sample_NER(sample_coef, batch_size, exp_replay, cur_pref, cur_state):
    if exp_replay.size() == 0:
        return []

    samples_size = min(int(sample_coef * batch_size), exp_replay.size())
    exp_list = []  # (experience, similarity)

    for _ in range(samples_size):
        sample_exp = exp_replay.sample()
        if sample_exp is None:
            continue
        state_sim_score, pref_sim_score = compute_similarity(cur_state, sample_exp.state, cur_pref, sample_exp.preference_vector)
        total_score = state_sim_score + pref_sim_score
        exp_list.append((sample_exp, total_score))

    ordered_exp_list = sorted(exp_list, key=lambda x: x[1])
    return ordered_exp_list[:samples_size]

def generate_random_preference_vector():
    """
    Generates a random preference vector of length 3,
    representing weights for:
    - Profit
    - energy saving
    - compute usage

    The vector components sum to exactly 1.
    """
    # Generate 3 random positive values
    raw_values = [random.random() for _ in range(3)]

    # Normalize so sum is 1
    total = sum(raw_values)
    normalized_vector = np.array([v / total for v in raw_values])
    return normalized_vector
