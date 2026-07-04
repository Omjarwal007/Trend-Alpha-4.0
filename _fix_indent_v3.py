filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix indentation: lines 1367-1395 (0-indexed 1366-1394) have 4 extra spaces
# due to broken try/except insert. Remove 4 leading spaces.
for i in range(1366, min(1395, len(lines))):
    if lines[i].startswith('                    '):
        lines[i] = lines[i][4:]  # remove 4 leading spaces

# Also fix line 1363 - the indented else body 'daily_ret = 0.0'
# It's at 24 spaces, should be at 20 (else is at 16, body should be +4 = 20)
for i in range(len(lines)):
    stripped = lines[i].lstrip()
    if stripped.startswith('daily_ret = 0.0') and lines[i].startswith('                        '):
        # Count leading spaces
        leading = len(lines[i]) - len(stripped)
        if leading == 24:
            lines[i] = '                    ' + stripped + '\n'
            break

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Indent fix applied")
