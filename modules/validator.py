# modules/validator.py
import numpy as np
import pandas as pd

def validate_simulation(users, bs):
    issues = []
    score = 100

    if users["x"].isna().any() or users["y"].isna().any():
        issues.append("❌ Missing user coordinates detected.")
        score -= 20
    if (bs["capacity"] <= 0).any():
        issues.append("⚠️ Invalid base station capacity (<= 0).")
        score -= 10
    if users["Connected_BS"].max() >= len(bs):
        issues.append("❌ Invalid Base Station index assigned.")
        score -= 25
    if len(users["Connected_BS"].unique()) < len(bs) / 3:
        issues.append("⚠️ Possible imbalance: most users connected to few towers.")
        score -= 10

    if score < 0:
        score = 0

    if not issues:
        issues.append("✅ Simulation data validated successfully.")
    return issues, max(score, 0)


def validate_metrics(metrics):
    issues = []
    score = 100

    if not 0 <= metrics["Jain_Index"] <= 1:
        issues.append("❌ Jain Index out of valid range (0–1).")
        score -= 25
    if metrics["Throughput"] < 0:
        issues.append("❌ Negative throughput detected.")
        score -= 25
    if metrics["Avg_Utilization"] > 1.5:
        issues.append("⚠️ Utilization appears unusually high (>100%).")
        score -= 10

    if score < 0:
        score = 0

    if not issues:
        issues.append("✅ Metrics values appear valid.")
    return issues, max(score, 0)


def validate_logs(df):
    issues = []
    score = 100

    if df.isna().any().any():
        issues.append("⚠️ Missing values in log entries.")
        score -= 10
    if df["Jain_Index"].between(0, 1).mean() < 1:
        issues.append("⚠️ Some Jain Index values outside range.")
        score -= 10
    if df["Throughput"].lt(0).any():
        issues.append("❌ Negative throughput in logs.")
        score -= 25

    if score < 0:
        score = 0

    if not issues:
        issues.append("✅ Log file validated successfully.")
    return issues, max(score, 0)


def check_stability(df):
    stability_report = []
    score = 100

    if "Jain_Index" in df:
        std_fair = df["Jain_Index"].std()
        if std_fair > 0.15:
            stability_report.append("⚠️ High fairness variance (unstable results).")
            score -= 15
        else:
            stability_report.append("✅ Fairness results stable.")
    if "Throughput" in df:
        std_thr = df["Throughput"].std()
        if std_thr > 30:
            stability_report.append("⚠️ Throughput fluctuates significantly.")
            score -= 10
        else:
            stability_report.append("✅ Throughput stable across tests.")

    return stability_report, max(score, 0)
