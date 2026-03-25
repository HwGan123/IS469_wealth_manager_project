import requests

url = "https://www.sec.gov/Archives/edgar/data/0000320193/000032019325000079/aapl-20250927.htm"
headers = {"User-Agent": "Junjie yujunjie1999@gmail.com"}  # Required by SEC

r = requests.get(url, headers=headers)
with open("aapl_10k_2025.htm", "wb") as f:
    f.write(r.content)
print("Downloaded!")