import json
import math
from collections import Counter

with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total structures:", len(data))
total_sites = sum(len(d["atoms"]["elements"]) for d in data)
print("Total sites:", total_sites)
elements = set(el for d in data for el in d["atoms"]["elements"])
print("Total elements:", len(elements))
chem_systems = set(tuple(sorted(set(d["atoms"]["elements"]))) for d in data)
print("Total chemical systems:", len(chem_systems))

def get_reduced_formula(elements):
    counts = Counter(elements)
    g = math.gcd(*counts.values())
    parts = []
    for el in sorted(counts.keys()):
        c = counts[el] // g
        parts.append(f"{el}{c if c > 1 else ''}")
    return "".join(parts)

formulas = set(get_reduced_formula(d["atoms"]["elements"]) for d in data)
print("Total reduced formulas:", len(formulas))
