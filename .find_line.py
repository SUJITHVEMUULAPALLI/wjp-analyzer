import re
p = r'src\\wjp_analyser\\web\\app.py'
with open(p, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'secret_key' in line:
            print(f"{p}:{i}")
            break
