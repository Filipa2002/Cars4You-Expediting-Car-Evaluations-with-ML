import requests
import os

API_KEY = "c2a876319341d86d00ef5bf2eed293b7" 
BASE = "https://api.wheel-size.com/v2"
OUT = "brands_models.csv"

done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        next(f)
        done = {line.split(",")[0] for line in f}

makes = requests.get(f"{BASE}/makes/", params={"user_key": API_KEY}).json()
makes = makes["data"] if isinstance(makes, dict) and "data" in makes else makes

with open(OUT, "a", encoding="utf-8") as f:
    if os.path.getsize(OUT) == 0: f.write("brand,model\n")
    new = 0
    for mk in makes:
        if mk["name"] in done: continue
        models = requests.get(f"{BASE}/models/", params={"make": mk["slug"], "user_key": API_KEY}).json()
        models = models["data"] if isinstance(models, dict) and "data" in models else models
        for m in models: f.write(f"{mk['name']},{m['name']}\n")
        new += 1
        if new >= 100: break
print("✅ Feito.")