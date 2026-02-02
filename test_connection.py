import requests
import json

# Define a standard User-Agent
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# --- Test 1: Public Endpoint ---
print("--- Running Test 1: Public Binance Endpoint ---")
public_url = "https://api.binance.com/api/v3/time"

try:
    response_public = requests.get(public_url, headers=headers, timeout=10)
    print(f"Status Code: {response_public.status_code}")
    print("Response Body:")

    # Try to print as JSON, fall back to raw text
    try:
        print(json.dumps(response_public.json(), indent=2))
    except json.JSONDecodeError:
        print(response_public.text)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

print("\n" + "=" * 50 + "\n")

# --- Test 2: What is my IP? ---
print("--- Running Test 2: Checking Public IP Address ---")
ip_check_url = "https://api.ipify.org?format=json"

try:
    response_ip = requests.get(ip_check_url, timeout=10)
    print(f"Status Code: {response_ip.status_code}")

    ip_data = response_ip.json()
    print(f"Server's Public IP appears to be: {ip_data['ip']}")
    print("^^^ Make SURE this IP is the one whitelisted on Binance! ^^^")

except requests.exceptions.RequestException as e:
    print(f"Could not determine public IP. An error occurred: {e}")
