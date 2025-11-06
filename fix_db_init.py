import re, os, io
p = r"src/wjp_analyser/database/__init__.py"
s = open(p, 'r', encoding='utf-8').read()
# Remove any PowerShell artifacts
s = re.sub(r"^.*System\.IO.*$\n", "", s, flags=re.M)
s = re.sub(r"^.*elseif.*$\n", "", s, flags=re.M)
s = re.sub(r"^\s*\}\s*$\n?", "", s, flags=re.M)
# Normalize common method names
s = s.replace('EndsWith(', 'endswith(').replace('os.path.Join', 'os.path.join')
# Ensure sqlite block is valid Python
s = re.sub(r"if db_type == \"sqlite\":[\s\S]*?database_url = .*?\n",
           """if db_type == \"sqlite\":
        # If name looks like a path, respect it; else place under data/<name>.db
        db_path = name
        try:
            if not os.path.isabs(db_path) and not db_path.endswith(\".db\"):
                db_path = os.path.join(\"data\", f\"{db_path}.db\")
            elif os.path.isabs(db_path) and not db_path.endswith(\".db\"):
                db_path = f\"{db_path}.db\"
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        except Exception:
            db_path = os.path.join(\"data\", \"wjp_analyser.db\")
        database_url = f\"sqlite:///{db_path}\"\n""",
           s, flags=re.M)
open(p, 'w', encoding='utf-8').write(s)
print('sanitized')