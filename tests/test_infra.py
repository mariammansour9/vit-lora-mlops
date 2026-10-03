"""Config sanity checks: alert rules, compose and Prometheus config must parse and agree."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_prometheus_and_alert_rules_parse():
    prom = yaml.safe_load((ROOT / "monitoring/prometheus.yml").read_text())
    jobs = {j["job_name"] for j in prom["scrape_configs"]}
    assert {"vit-api", "node", "cadvisor"} <= jobs
    rules = yaml.safe_load((ROOT / "monitoring/alerts.yml").read_text())
    names = {r["alert"] for g in rules["groups"] for r in g["rules"]}
    assert {"DataDriftHigh", "ModelAccuracyLow", "HighCPU", "HighMemory", "ApiDown"} <= names


def test_compose_services():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert {"api", "prometheus", "grafana", "node-exporter", "cadvisor"} <= set(c["services"])


def test_dashboard_json_valid():
    import json
    d = json.loads((ROOT / "monitoring/grafana/dashboards/vit.json").read_text())
    assert len(d["panels"]) >= 8
