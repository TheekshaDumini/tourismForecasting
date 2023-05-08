from utils.data import load_dataset
from utils.infer import load_model, get_counts
import yaml
import numpy as np

ds = load_dataset("./data")
model = load_model("./model")
with open("./data/config.yaml") as handler:
    raw_countries = yaml.load(handler, yaml.FullLoader)["countries"]


def predict(event):
    required_keys = ["countries", "month", "year", "n_months"]
    request_ok = all([k in required_keys for k in event.keys()])
    if request_ok:
        countries = event["countries"]
        month = event["month"]
        year = event["year"]
        n_months = event["n_months"]
        body = {}
        for country in countries:
            if country == "ALL":
                counts = get_counts("INDIA", month, year, n_months, ds, model)
                counts["count"] = (np.array(counts["count"]) * 5.1234).astype(int).tolist()
            else:
                counts = get_counts(country, month, year, n_months, ds, model)
            body[country] = counts

        return {
            "status": 200,
            "headers": {"content-type": "json"},
            "body": body,
        }
    else:
        return {
            "status": 400,
            "headers": {"content-type": "json"},
            "body": {
                "message": f"Required keys are not present. Required keys are {required_keys}"
            },
        }


def get_metadata():
    countries = [*raw_countries]
    countries.append("ALL")
    body = {
        "countries": countries,
        "date_range": {
            "available": {
                "start": {"year": 2018, "month": "January"},
                "end": {"year": 2022, "month": "December"},
            },
            "predictable": {
                "start": {"year": 2022, "month": "January"},
                "end": {"year": 2022, "month": "February"},
            },
        },
    }
    return {
        "status": 200,
        "headers": {"content-type": "json"},
        "body": body,
    }


def get_stats(year):
    df = ds.df
    rel_recs = df[[dt.year == year for dt in df.index]]
    counts = {}
    scale, offset = (
        ds.normalize_para["scale"]["count"],
        ds.normalize_para["offset"]["count"],
    )
    for country in raw_countries:
        counts[country] = int(
            np.round(rel_recs[rel_recs[country] == 1]["count"].sum() * scale + offset)
        )

    return {
        "status": 200,
        "headers": {"content-type": "json"},
        "body": {"counts": counts},
    }
