import requests
import time

BASE_URL = "https://mara-hackathon-api.onrender.com"
api_keys = [
    "e09febc0-35a0-4e9b-9ae8-65290e27949a",
    "247485c6-6ba4-41b2-8936-8d7fc025df10",
    "1ad52e93-4a7a-4b0a-8efc-a123078f2ee9",
    "f563f8b2-319c-4a6c-a910-1f265b9d2d0a",
    "474c7961-6b47-4b71-b3cf-a5bafcfb953e"
]

def set_initial_allocation(api_key):
    headers = {"X-API-Key": api_key}
    allocation_payload = {
        "asic_compute": 5,
        "gpu_compute": 10,
        "immersion_miners": 5,
        "air_miners": 0,
        "hydro_miners": 0
    }
    try:
        response = requests.put(f"{BASE_URL}/machines", json=allocation_payload, headers=headers)
        response.raise_for_status()
        print(f"[OK] Allocation successful for {api_key}")
    except requests.HTTPError as e:
        print(f"[ERR] Failed to allocate machines for {api_key}: {e}")

def fetch_site_info(api_key):
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{BASE_URL}/machines", headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    # Step 1: Allocate machines once
    for key in api_keys:
        set_initial_allocation(key)

    print("\n--- Entering polling mode: checking site status every 10 seconds ---\n")

    # Step 2: Repeatedly fetch site status every 10 seconds
    while True:
        for key in api_keys:
            try:
                site_info = fetch_site_info(key)
                print(f"Site info for API key {key}:")
                print(site_info)
                print("-" * 60)
            except requests.HTTPError as e:
                print(f"[ERR] Failed to fetch site info for {key}: {e}")
        time.sleep(10)

if __name__ == "__main__":
    main()
