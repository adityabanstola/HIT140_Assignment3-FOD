# HIT140 Assignment 3 — CSV Builder

## Group Members

| Name | Student ID |
|---|---|
| Ashim Koirala | S407089 |
| Aadesh Bhandari  | S404328  |
| Aditya Banstola  | S403120  |
| Sandesh Regmi  | S407215  |

## Overview

Run `collect_data.py` to download and process the configured public data sources and generate the CSV files for the assignment.

## Generated Files

- `world_cup_2026_matches.csv` — expected 104 match rows
- `model_2_1_goal_difference.csv` — expected 104 rows with 8 predictors
- `model_2_2_goals_scored.csv` — expected 208 rows with 8 predictors
- `data_quality_report.txt` — data quality checks and predictor details

Expected row counts may vary if source data or processing changes. Review the quality report before modelling.

## Requirements

- Python 3
- `pandas`
- `requests`

## Run on Windows

Open the project folder in VS Code, select **Terminal → New Terminal**, then run:

```bash
py -m pip install pandas requests
py collect_data.py
```

Review `data_quality_report.txt` after the script finishes.

## Model Outputs

- **Model 2.1:** Target variable `goal_difference`; eight predictors documented in `collect_data.py` and `data_quality_report.txt`.
- **Model 2.2:** Target variable `goals_scored`; eight predictors documented in `collect_data.py` and `data_quality_report.txt`.

## Data Validation

- Check missing values and the `*_history_matches_used` columns.
- Verify match scores against official FIFA results where available.
- Predictors should use only information available before each fixture. Same-day historical matches are excluded to reduce data leakage.
- Do not use a match's final score or other post-kick-off information as a predictor.
- Do not invent missing values if a source or validation check fails.
- Ensure all group members understand and validate the data and calculations, and disclose AI assistance in accordance with CDU requirements.

## Data Sources

- [OpenFootball — 2026 World Cup data](https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.json)
- [International match history](https://github.com/martj42/international_results)
- [FIFA — tournament statistics](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics)
- [FIFA — fixtures and results](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums)
- [April 2026 FIFA ranking report](https://www.espn.co.uk/football/story/_/id/46664763/fifa-mens-top-50-world-rankings)
