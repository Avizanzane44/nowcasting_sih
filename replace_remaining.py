import re

with open('dashboard/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace line 158
text = re.sub(r'<option value="delhi_ncr">[^\x00-\x7F]*\s*Delhi-NCR', '<option value="delhi_ncr">📍 Delhi-NCR', text)

# Replace line 159
text = re.sub(r'<option value="himalayan_belt">[^\x00-\x7F]*\s*Himalayan Belt', '<option value="himalayan_belt">🏔️ Himalayan Belt', text)

# Replace line 242
text = re.sub(r'[^\x00-\x7F]*\s*Earth Satellite', '🛰️ Earth Satellite', text)

# Replace line 248
text = re.sub(r'[^\x00-\x7F]*\s*Street Nav', '🛣️ Street Nav', text)

with open('dashboard/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("Remaining bad chars replaced.")
