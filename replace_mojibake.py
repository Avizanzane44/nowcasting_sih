import re

with open('dashboard/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    '1â€“3 KM': '1–3 KM',
    'â€¢': '•',
    'â€“': '–',
    'ðŸ“ ': '📍',
    'ðŸ”=': '📍',
    'ðŸ ”ï¸ ': '🏔️',
    'ðŸŒŠ': '🌊',
    'âš¡': '⚡',
    'ðŸŒ™': '🌙',
    'ðŸ›°ï¸ ': '🌍',
    'ðŸ—ºï¸ ': '🗺️',
    'ðŸ ™ï¸ ': '🛣️',
    'ðŸŒ': '🌍',
    'ðŸ—º': '🗺️',
    'ðŸ›£': '🛣️',
}

for bad, good in replacements.items():
    text = text.replace(bad, good)

# Clean up any remaining bad characters starting with â€ or ðŸ
# First, let's find all occurrences to see what else we missed
import sys
for match in set(re.findall(r'â€.', text)):
    print(f"Remaining â€: {repr(match)}")
for match in set(re.findall(r'ðŸ..?', text)):
    print(f"Remaining ðŸ: {repr(match)}")

with open('dashboard/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("Replacement complete.")
