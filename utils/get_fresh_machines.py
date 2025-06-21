import requests

BASE_URL = "https://mara-hackathon-api.onrender.com"

def create_site(name):
    response = requests.post(
        f"{BASE_URL}/sites",
        json={"name": name}
    )
    response.raise_for_status()
    return response.json()

def main():
    api_keys = []
    for i in range(5):
        site_name = f"autogen-site-{i}"
        try:
            data = create_site(site_name)
            api_key = data["api_key"]
            print(f"[{site_name}] API Key: {api_key}")
            api_keys.append(api_key)
        except requests.HTTPError as e:
            print(f"Failed to create site {site_name}: {e}")
    print("\nAll API keys:")
    for key in api_keys:
        print(key)

if __name__ == "__main__":
    main()
