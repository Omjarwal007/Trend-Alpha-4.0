"""Comprehensive structural fix for dashboard.py"""

filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = 0

# Pattern 1: Find 'else:' at wrong indentation relative to its 'if'
i = 0
while i < len(lines):
    stripped = lines[i].rstrip()
    if stripped == 'else:':
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        # Look back for matching if at higher indent
        for j in range(i-1, max(0, i-10), -1):
            prev = lines[j]
            prev_stripped = prev.rstrip()
            prev_indent = len(prev) - len(prev.lstrip())
            # if/elif that should match this else
            if prev_indent > current_indent and (prev_stripped.endswith(':') and 
                ('if ' in prev_stripped.split('#')[0] or 'elif ' in prev_stripped.split('#')[0])):
                # This else should be at the same indent as the if/elif
                lines[i] = ' ' * prev_indent + 'else:\n'
                fixed += 1
                break
    i += 1

# Pattern 2: Fix lines after else that have wrong indentation
# If an else: was just fixed, the next non-blank line should be at else_indent + 4
i = 0
while i < len(lines) - 1:
    stripped = lines[i].rstrip()
    if stripped == 'else:':
        else_indent = len(lines[i]) - len(lines[i].lstrip())
        # Find next non-blank line
        for j in range(i+1, min(i+20, len(lines))):
            next_stripped = lines[j].rstrip()
            if next_stripped and not next_stripped.strip().startswith('#'):
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if next_indent <= else_indent:
                    # This line should be inside the else body (indent + 4)
                    lines[j] = ' ' * (else_indent + 4) + lines[j].lstrip()
                    fixed += 1
                elif next_indent != else_indent + 4 and next_indent > else_indent:
                    # Wrong indent, fix it
                    pass  # Too complex to fix automatically
                break
    i += 1

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Fixed {fixed} structural issues')

import py_compile
try:
    py_compile.compile(filepath, doraise=True)
    print('✅ FILE CLEAN')
except py_compile.PyCompileError as e:
    print(f'❌ ERROR at line {e.lineno}: {e.msg}')
