import requests

url = "https://api.openalex.org/works"

params = {
    "filter": "from_publication_date:2020-01-01,to_publication_date:2026-12-31,has_abstract:true",
    "per-page": 1,
    "mailto": "n12028114@students.meduniwien.ac.at",
}

r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
data = r.json()

print("Total OpenAlex works with abstract, 2020-2026:")
print(data["meta"]["count"])

import requests

MAILTO = "n12028114@students.meduniwien.ac.at"
BASE_FILTER = "from_publication_date:2020-01-01,to_publication_date:2026-12-31,has_abstract:true"

queries = {
    "all_with_abstract": None,
    "engineering_general": "engineering",
    "mechanical_engineering": '"mechanical engineering"',
    "materials_engineering": '"materials engineering"',
    "soft_robotics": '"soft robotics"',
    "microfluidics": "microfluidics",
    "hydrogel": "hydrogel",
    "pressure_flow_control": '"pressure control" fluid',
    "sensors": '"pressure sensor" OR "force sensor"',
    "sealing_leakage": '"leak sealing" OR "self sealing"',
    "bioadhesive_hydrogel": '"bioadhesive hydrogel"',
    "continuum_robot": '"continuum robot"',
}

for name, search in queries.items():
    params = {
        "filter": BASE_FILTER,
        "per-page": 1,
        "mailto": MAILTO,
    }
    if search:
        params["search"] = search

    r = requests.get("https://api.openalex.org/works", params=params, timeout=60)
    r.raise_for_status()
    count = r.json()["meta"]["count"]
    print(f"{name}: {count:,}")