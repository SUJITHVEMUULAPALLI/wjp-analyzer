import sys
sys.path.insert(0, r"C:\WJP ANALYSER")
import app as m
c = m.app.test_client()
r = c.get("/health")
print("status", r.status_code)
print(r.json)
