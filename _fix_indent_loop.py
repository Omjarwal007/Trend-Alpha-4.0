"""Auto-fix ALL indentation errors in a loop."""
import sys

filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

for iteration in range(200):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, filepath, 'exec')
        print(f"✅ FILE CLEAN after {iteration} fixes")
        sys.exit(0)
    except SyntaxError as e:
        lines = content.split('\n')
        lineno = e.lineno - 1
        if lineno < 0 or lineno >= len(lines):
            print(f"❌ Invalid line")
            sys.exit(1)
        
        bad = lines[lineno]
        stripped = bad.lstrip()
        current_indent = len(bad) - len(stripped)
        
        # Try removing 4 spaces if it's a dedent error
        if current_indent >= 4:
            lines[lineno] = bad[4:]
            content = '\n'.join(lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            continue
        
        # Try adding 4 spaces if it's an indent error  
        lines[lineno] = '    ' + bad
        content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print(f"⚠️ Hit max iterations")
sys.exit(1)
