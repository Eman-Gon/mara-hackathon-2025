import requests
import json
BASE_URL = "https://mara-hackathon-api.onrender.com"
api_keys = [
    "e09febc0-35a0-4e9b-9ae8-65290e27949a",
    "247485c6-6ba4-41b2-8936-8d7fc025df10",
    "1ad52e93-4a7a-4b0a-8efc-a123078f2ee9",
    "f563f8b2-319c-4a6c-a910-1f265b9d2d0a",
    "474c7961-6b47-4b71-b3cf-a5bafcfb953e"
]

def fetch_site_info(api_key):
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{BASE_URL}/machines", headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    for key in api_keys:
        try:
            site_info = fetch_site_info(key)
            print(f"Site info for API key {key}:")
            print(json.dumps(site_info, indent=2))
            print("-" * 60)
        except requests.HTTPError as e:
            print(f"Failed to fetch site info for {key}: {e}")

if __name__ == "__main__":
    main()