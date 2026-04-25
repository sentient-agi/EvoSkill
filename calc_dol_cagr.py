import math

# FY 2011 Department of Labor total outlays (ACTUAL from Treasury Bulletin FFO-3 table)
# Line in treasury_bulletin_2012_09.txt shows: | 2011 | ... | 131973 |
fy_2011 = 131973  # in millions

# FY 2019 Department of Labor total outlays (actual from Treasury Bulletin FFO-3 table)
# Line in treasury_bulletin_2020_09.txt shows: | 2019 | ... | 35810 |
fy_2019 = 35810  # in millions

# Number of years from FY 2011 to FY 2019
n = 8

# Calculate CAGR
cagr = (fy_2019 / fy_2011) ** (1/n) - 1

# Calculate annual decay factor
decay_factor = 1 + cagr

# Calculate arc elasticity using midpoint formula
# Arc elasticity = (ΔQ/ΔP) × ((P1+P2)/(Q1+Q2))
# where Q is quantity (outlays), P is time period index
# P1 = 1 (FY 2011), P2 = 9 (FY 2019)

Q1 = float(fy_2011)
Q2 = float(fy_2019)
P1 = 1.0
P2 = float(n + 1)  # 9

# Arc elasticity using midpoint percentage change formula
pct_change_Q = (Q2 - Q1) / ((Q1 + Q2) / 2)
pct_change_P = (P2 - P1) / ((P1 + P2) / 2)

arc_elasticity = pct_change_Q / pct_change_P

print(f"FY 2011 DOL outlays (actual): {fy_2011} million")
print(f"FY 2019 DOL outlays (actual): {fy_2019} million")
print(f"Number of years: {n}")
print()
print(f"CAGR: {cagr}")
print(f"CAGR (rounded to 3 decimal places): {round(cagr, 3)}")
print()
print(f"Annual decay factor: {decay_factor}")
print(f"Annual decay factor (rounded to 3 decimal places): {round(decay_factor, 3)}")
print()
print(f"Q1={Q1}, Q2={Q2}")
print(f"P1={P1}, P2={P2}")
print(f"% change in Q (midpoint): {pct_change_Q}")
print(f"% change in P (midpoint): {pct_change_P}")
print(f"Arc elasticity: {arc_elasticity}")
print(f"Arc elasticity (rounded to 3 decimal places): {round(arc_elasticity, 3)}")
print()
print("Final Answer: [{}, {}, {}]".format(
    round(cagr, 3),
    round(decay_factor, 3),
    round(arc_elasticity, 3)
))
