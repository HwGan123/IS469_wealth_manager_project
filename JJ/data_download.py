import requests
from bs4 import BeautifulSoup

url = "https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm"
headers = {"User-Agent": "Junjie yujunjie1999@gmail.com"}

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, "html.parser")

# Remove script and style tags entirely
for tag in soup(["script", "style"]):
    tag.decompose()

# Extract plain text
text = soup.get_text(separator="\n", strip=True)

with open("nvda_10k_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Downloaded!")