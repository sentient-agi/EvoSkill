import re
import math
import os

base_dir = "/mnt/data/treasury_bulletins_parsed"

def extract_dol_total_outlays(filepath):
    """Extract Department of Labor total outlays (including budgetary and trust-fund flows)"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find the section with "Department of Labor" that shows budgetary and trust fund flows
    # This appears to be a section with multiple columns showing different fund types
    in_section = False
    for i, line in enumerate(lines):
        # Look for the section with DoL and trust fund data
        if 'Department of Labor' in line and i > 100:
            # Check if this is the section with budgetary/trust fund breakdown
            # This section typically has format like:
            # | Department of Labor | value1 | value2 | value3 | total |
            # We want the total that includes both budgetary and trust-fund flows
            
            # Parse the line to get values
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p and p != 'nan']
            
            # Find numeric values in the line
            nums = []
            for p in parts:
                p = p.replace(',', '')
                # Remove any alphabetic annotations
                p_clean = re.sub(r'[a-zA-Z\r]', '', p).strip()
                if p_clean and p_clean not in ['', '-']:
                    try:
                        nums.append(float(p_clean))
                    except:
                        pass
            
            if nums:
                # The total outlays including budgetary and trust-fund flows 
                # should be the sum or the last/largest value
                # Based on the pattern, the format appears to be:
                # budgetary | trust | other adjustments | total
                return nums
            break
    
    return None

def extract_fy_total(bulletin_path, fy_year):
    """Extract total outlays for a full fiscal year"""
    # For FY 2011, we need to sum Oct 2010 through Sept 2011
    # The data for each month appears in the following month's bulletin
    # October 2010 data appears in November 2010 bulletin (or later)
    # But a cleaner approach is to find the yearly summary line
    
    with open(bulletin_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Look for the fiscal year annual total line
    # Pattern: | 2011 | ... values ... |
    for line in lines:
        if f'^{fy_year} |' in line or f'^{fy_year} -' in line:
            continue
        # Look for exact match of fiscal year (no Est suffix)
        pattern = rf'^\|\s*{fy_year}\s*\|'
        if re.search(pattern, line):
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p and p != 'nan']
            
            nums = []
            for p in parts:
                p = p.replace(',', '')
                p_clean = re.sub(r'[a-zA-Z*r]', '', p).strip()
                if p_clean and p_clean not in ['', '-']:
                    try:
                        nums.append(float(p_clean))
                    except:
                        pass
            
            if nums:
                return nums
    return None

# First, let's examine the structure of the data more carefully
# Look at treasury_bulletin_2011_09 for FY 2011 data
test_file = os.path.join(base_dir, "treasury_bulletin_2011_09.txt")

# Also look at treasury_bulletin_2020_09 for FY 2019 data  
test_file_2019 = os.path.join(base_dir, "treasury_bulletin_2020_09.txt")

print("Examining FY 2011 data in treasury_bulletin_2011_09.txt...")
with open(test_file, 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Department of Labor' in line and i > 200 and i < 600:
        print(f"Line {i}: {line[:200]}")

print("\n" + "="*80)
print("Examining FY 2019 data in treasury_bulletin_2020_09.txt...")
with open(test_file_2019, 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Department of Labor' in line and i > 200 and i < 600:
        print(f"Line {i}: {line[:200]}")
