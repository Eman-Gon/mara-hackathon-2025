import os
import time
import itertools
import json
import requests

BASE_URL = "https://mara-hackathon-api.onrender.com"
SESSION = requests.Session()

def create_site(name: str):
    """POST /sites → returns {api_key, name, power, ...}"""
    r = SESSION.post(f"{BASE_URL}/sites", json={"name": name})
    r.raise_for_status()
    return r.json()

def set_api_key(key: str):
    SESSION.headers.update({"X-Api-Key": key})

def get_prices():
    """GET /prices → returns list of {energy_price, hash_price, token_price, timestamp}"""
    r = SESSION.get(f"{BASE_URL}/prices")
    r.raise_for_status()
    return r.json()

def get_inventory():
    """GET /inventory → returns the static inventory spec"""
    r = SESSION.get(f"{BASE_URL}/inventory")
    r.raise_for_status()
    return r.json()

def allocate_machines(allocation: dict):
    """PUT /machines → returns the new allocation & metrics"""
    r = SESSION.put(f"{BASE_URL}/machines", json=allocation)
    r.raise_for_status()
    return r.json()

def get_machines():
    """GET /machines → returns current site status including revenue, power used, etc."""
    r = SESSION.get(f"{BASE_URL}/machines")
    r.raise_for_status()
    return r.json()

def generate_allocations(inv: dict, max_power: int):
    """
    Simple greedy generator:
    - Fill asic_compute until power exhausted
    - Then gpu_compute
    - Then immersion_miners (or any other), etc.
    Yields a handful of candidate dicts.
    """
    opts = []
    # per-unit power cost from inventory
    p_asic = inv["inference"]["asic"]["power"]
    p_gpu   = inv["inference"]["gpu"]["power"]
    p_imm   = inv["miners"]["immersion"]["power"]
    # max possible counts
    max_a = max_power // p_asic
    max_g = max_power // p_gpu
    max_i = max_power // p_imm

    # for demo, just three simple combos
    opts.append({"asic_compute": max_a, "gpu_compute": 0,       "immersion_miners": 0})
    opts.append({"asic_compute": 0,     "gpu_compute": max_g,   "immersion_miners": 0})
    opts.append({"asic_compute": 0,     "gpu_compute": 0,       "immersion_miners": max_i})
    return opts

def collect_data(site_name: str, n_steps: int = 10, out_path="experience.jsonl"):
    """
    High-level loop: for a single site, sample allocations,
    call the API, and log (state, action, reward) for each step.
    """
    site = create_site(site_name)
    set_api_key(site["api_key"])
    max_power = site["power"]

    inv    = get_inventory()
    prices = get_prices()

    with open(out_path, "a") as f:
        for step in range(n_steps):
            # state: inventory + latest prices + remaining power
            state = {
                "timestamp": time.time(),
                "prices": prices[-1],   # most recent
                "inventory": inv,
                "site_power": max_power,
            }
            # choose a candidate allocation (you can swap in your own strategy here)
            for alloc in generate_allocations(inv, max_power):
                action = {
                    "asic_compute": alloc.get("asic_compute", 0),
                    "gpu_compute":   alloc.get("gpu_compute", 0),
                    "immersion_miners": alloc.get("immersion_miners", 0),
                }
                # apply allocation
                machines_resp = allocate_machines(action)
                # observe new state / metrics
                status = get_machines()
                # define a simple scalar reward: e.g. revenue - power_cost
                reward = status["total_revenue"] - status["total_power_cost"]
                # log one experience tuple
                experience = {
                    "state":  state,
                    "action": action,
                    "next_state": status,
                    "reward": reward,
                }
                f.write(json.dumps(experience) + "\n")
                print(f"Step {step}, alloc={action}, reward={reward:.2f}")
            # re-fetch prices for next iteration
            prices = get_prices()

if __name__ == "__main__":
    collect_data("MyTrainingSite", n_steps=5)
