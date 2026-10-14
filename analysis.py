"""
HIT140 Assignment 3 — Objective 2
Two linear regression models and three visualisations.

Required CSV files in the same folder as this script:
- world_cup_2026_matches.csv
- model_2_1_goal_difference.csv
- model_2_2_goals_scored.csv

Run with:
    python analysis.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. LOAD DATA
folder = Path(__file__).resolve().parent

matches_path = folder / "world_cup_2026_matches.csv"
model21_path = folder / "model_2_1_goal_difference.csv"
model22_path = folder / "model_2_2_goals_scored.csv"

for path in (matches_path, model21_path, model22_path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path.name}\n"
            "Place this CSV in the same folder as analysis.py."
        )

matches = pd.read_csv(matches_path)
model21 = pd.read_csv(model21_path)
model22 = pd.read_csv(model22_path)

print("=" * 65)
print("HIT140 ASSIGNMENT 3 — OBJECTIVE 2")
print("=" * 65)

print("\nMATCH DATA")
print("Shape:", matches.shape)
print(matches.head().to_string(index=False))

print("\nMODEL 2.1 — GOAL DIFFERENCE")
print("Shape:", model21.shape)
print(model21.head().to_string(index=False))

print("\nMODEL 2.2 — GOALS SCORED")
print("Shape:", model22.shape)
print(model22.head().to_string(index=False))

# ------------------------------------------------------------
# 2. DEFINE AND VALIDATE PREDICTORS
# ------------------------------------------------------------
predictors_21 = [
    "team_a_fifa_rank",
    "team_b_fifa_rank",
    "team_a_win_rate_last5",
    "team_b_win_rate_last5",
    "team_a_avg_goals_scored_last5",
    "team_b_avg_goals_scored_last5",
    "team_a_avg_goals_conceded_last5",
    "team_b_avg_goals_conceded_last5",
]

predictors_22 = [
    "team_avg_goal_difference_last5",
    "opponent_avg_goal_difference_last5",
    "team_last_match_goals_scored",
    "opponent_last_match_goals_scored",
    "team_clean_sheet_rate_last5",
    "opponent_clean_sheet_rate_last5",
    "team_loss_rate_last5",
    "opponent_loss_rate_last5",
]

required21 = predictors_21 + ["match_id", "goal_difference", "team_a_goals", "team_b_goals"]
required22 = predictors_22 + [
    "match_id", "goals_scored", "goals_conceded",
    "team_history_matches_used", "opponent_history_matches_used"
]

missing21 = sorted(set(required21) - set(model21.columns))
missing22 = sorted(set(required22) - set(model22.columns))
if missing21 or missing22:
    raise ValueError(
        f"Missing columns. Model 2.1: {missing21}; Model 2.2: {missing22}"
    )

print("\nPREDICTOR VALIDATION")
print("Model 2.1 predictor count:", len(predictors_21))
print("Model 2.2 predictor count:", len(predictors_22))
print("Model 2.1 missing predictor columns:", set(predictors_21) - set(model21.columns))
print("Model 2.2 missing predictor columns:", set(predictors_22) - set(model22.columns))
shared = set(predictors_21) & set(predictors_22)
print("Shared predictors:", shared)
print("Number of shared predictors:", len(shared))

if len(predictors_21) != 8 or len(predictors_22) != 8:
    raise ValueError("Each model must use exactly eight predictors.")
if len(shared) > 4:
    raise ValueError("The models share more than four predictors.")

# ------------------------------------------------------------
# 3. TARGET, ROW, SCORE, HISTORY AND MISSING-VALUE CHECKS
# ------------------------------------------------------------
print("\nTARGET AND DATA QUALITY CHECKS")
goal_difference_correct = (
    model21["goal_difference"]
    == model21["team_a_goals"] - model21["team_b_goals"]
).all()
print("Model 2.1 goal difference matches scores:", goal_difference_correct)

rows_per_match = model22.groupby("match_id").size()
print("Model 2.2 rows per match counts:", rows_per_match.value_counts().to_dict())

def scores_match_within_each_match(group):
    if len(group) != 2:
        return False
    return (
        group["goals_scored"].iloc[0] == group["goals_conceded"].iloc[1]
        and group["goals_scored"].iloc[1] == group["goals_conceded"].iloc[0]
    )

score_check = model22.groupby("match_id").apply(scores_match_within_each_match)
print("Model 2.2 scores are consistent within every match:", bool(score_check.all()))
print(
    "All goals are non-negative:",
    bool(
        model22["goals_scored"].ge(0).all()
        and model22["goals_conceded"].ge(0).all()
    ),
)

missing21_count = int(model21[predictors_21].isna().sum().sum())
missing22_count = int(model22[predictors_22].isna().sum().sum())
print("Missing predictor values in Model 2.1:", missing21_count)
print("Missing predictor values in Model 2.2:", missing22_count)

history_columns = [
    "team_a_history_matches_used",
    "team_b_history_matches_used",
]
history_columns_22 = [
    "team_history_matches_used",
    "opponent_history_matches_used",
]
print("\nHISTORY COUNTS — MODEL 2.1")
print(model21[history_columns].describe().round(3))
print("\nHISTORY COUNTS — MODEL 2.2")
print(model22[history_columns_22].describe().round(3))

if not goal_difference_correct:
    raise ValueError("Model 2.1 target check failed.")
if not (rows_per_match == 2).all():
    raise ValueError("Model 2.2 should contain exactly two rows per match.")
if not score_check.all():
    raise ValueError("Model 2.2 score consistency check failed.")
if missing21_count or missing22_count:
    raise ValueError("Missing predictor values found. Resolve them before modelling.")
if not (model22["goals_scored"].ge(0).all() and model22["goals_conceded"].ge(0).all()):
    raise ValueError("Negative goal values found.")

print("\nMODEL 2.1 FEATURE SUMMARY")
print(model21[predictors_21].describe().round(3))
print("\nMODEL 2.2 FEATURE SUMMARY")
print(model22[predictors_22].describe().round(3))

# 4. TRAIN AND EVALUATE MODEL 2.1
# GroupShuffleSplit keeps each match entirely in train or test.
# ------------------------------------------------------------
X21 = model21[predictors_21]
y21 = model21["goal_difference"]

split21 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train21, test21 = next(
    split21.split(X21, y21, groups=model21["match_id"])
)

model_21 = LinearRegression()
model_21.fit(X21.iloc[train21], y21.iloc[train21])
pred21 = model_21.predict(X21.iloc[test21])

mae21 = mean_absolute_error(y21.iloc[test21], pred21)
rmse21 = np.sqrt(mean_squared_error(y21.iloc[test21], pred21))


# 5. TRAIN AND EVALUATE MODEL 2.2
# Grouping prevents the two rows for a match being split across sets.
# ------------------------------------------------------------
X22 = model22[predictors_22]
y22 = model22["goals_scored"]

split22 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train22, test22 = next(
    split22.split(X22, y22, groups=model22["match_id"])
)

model_22 = LinearRegression()
model_22.fit(X22.iloc[train22], y22.iloc[train22])
pred22 = model_22.predict(X22.iloc[test22])

mae22 = mean_absolute_error(y22.iloc[test22], pred22)
rmse22 = np.sqrt(mean_squared_error(y22.iloc[test22], pred22))

print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)
print("Model 2.1 — Goal difference")
print("Training rows:", len(train21), "| Test rows:", len(test21))
print("MAE:", round(mae21, 3))
print("RMSE:", round(rmse21, 3))
print("\nModel 2.2 — Goals scored")
print("Training rows:", len(train22), "| Test rows:", len(test22))
print("MAE:", round(mae22, 3))
print("RMSE:", round(rmse22, 3))

# Save a concise metrics file for use when writing the report.
metrics = pd.DataFrame([
    {"model": "Model 2.1", "target": "Goal difference", "test_rows": len(test21),
     "MAE": mae21, "RMSE": rmse21},
    {"model": "Model 2.2", "target": "Goals scored", "test_rows": len(test22),
     "MAE": mae22, "RMSE": rmse22},
])
metrics.to_csv(folder / "model_evaluation_metrics.csv", index=False)


# 6. VISUALISATION 1 — MODEL 2.1 ACTUAL VS PREDICTED
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y21.iloc[test21], pred21, alpha=0.75)
low21 = min(float(y21.iloc[test21].min()), float(pred21.min()))
high21 = max(float(y21.iloc[test21].max()), float(pred21.max()))
ax.plot([low21, high21], [low21, high21], linestyle="--")
ax.set_xlabel("Actual goal difference")
ax.set_ylabel("Predicted goal difference")
ax.set_title("Model 2.1: Actual vs Predicted")
fig.tight_layout()
fig.savefig(folder / "model21_actual_vs_predicted.png", dpi=300)
plt.show()
plt.close(fig)


# 7. VISUALISATION 2 — MODEL 2.2 ACTUAL VS PREDICTED
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y22.iloc[test22], pred22, alpha=0.75)
low22 = min(float(y22.iloc[test22].min()), float(pred22.min()))
high22 = max(float(y22.iloc[test22].max()), float(pred22.max()))
ax.plot([low22, high22], [low22, high22], linestyle="--")
ax.set_xlabel("Actual goals scored")
ax.set_ylabel("Predicted goals scored")
ax.set_title("Model 2.2: Actual vs Predicted")
fig.tight_layout()
fig.savefig(folder / "model22_actual_vs_predicted.png", dpi=300)
plt.show()
plt.close(fig)


# 8. VISUALISATION 3 — STANDARDISED COEFFICIENT MAGNITUDES
# Standardise using training data only so coefficient magnitudes
# are more comparable across predictors measured on different scales.
# These are still coefficient magnitudes, not causal importance.
# ------------------------------------------------------------
standardised_21 = make_pipeline(StandardScaler(), LinearRegression())
standardised_21.fit(X21.iloc[train21], y21.iloc[train21])
coef21 = standardised_21.named_steps["linearregression"].coef_

standardised_22 = make_pipeline(StandardScaler(), LinearRegression())
standardised_22.fit(X22.iloc[train22], y22.iloc[train22])
coef22 = standardised_22.named_steps["linearregression"].coef_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, coefficients, predictors, title in [
    (axes[0], coef21, predictors_21, "Model 2.1"),
    (axes[1], coef22, predictors_22, "Model 2.2"),
]:
    coefficient_series = pd.Series(
        np.abs(coefficients), index=predictors
    ).sort_values()
    coefficient_series.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Absolute standardised coefficient")
fig.suptitle("Standardised Regression Coefficient Magnitudes")
fig.tight_layout()
fig.savefig(folder / "model_coefficients.png", dpi=300)
plt.show()
plt.close(fig)

print("\nSaved output files:")
print("- model21_actual_vs_predicted.png")
print("- model22_actual_vs_predicted.png")
print("- model_coefficients.png")
print("- model_evaluation_metrics.csv")
print("\nNote: Evaluate model performance alongside the data source, predictor timing,")
print("and assignment requirements. A successful run alone does not verify those items.")