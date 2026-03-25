from bs4 import BeautifulSoup
import re

with open("aapl_10k_2025.htm", "rb") as f:
    soup = BeautifulSoup(f, "html.parser")

# Remove iXBRL tags and noise
for tag in soup(["script", "style", "ix:header", "ix:hidden", "ix:nonfraction", "ix:nonNumeric"]):
    tag.decompose()

text = soup.get_text(separator="\n")
text = re.sub(r'\n{3,}', '\n\n', text).strip()

with open("aapl_10k_clean.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Extracted {len(text):,} characters")