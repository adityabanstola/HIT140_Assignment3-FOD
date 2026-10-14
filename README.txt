HIT140 Assignment 3 — CSV Builder

WHAT THIS BUILDER CREATES
When you run collect_data.py, it downloads its public source data and creates:
1. world_cup_2026_matches.csv                 (expected 104 match rows)
2. model_2_1_goal_difference.csv              (expected 104 rows; 8 predictors)
3. model_2_2_goals_scored.csv                 (expected 208 rows; 8 predictors)
4. data_quality_report.txt

HOW TO RUN ON WINDOWS (VS CODE)
1. Extract this ZIP into a folder.
2. Open that folder in VS Code.
3. Select Terminal > New Terminal.
4. Run:
   py -m pip install pandas requests
   py collect_data.py

IMPORTANT DATA CHECKS
- This package is a builder, not a guarantee that every source value is correct.
- It relies on public repositories and a pre-tournament ranking snapshot.
- Review data_quality_report.txt before modelling.
- Check missing values and the *_history_matches_used columns.
- Verify match scores against official FIFA results before submitting.
- The model predictors are calculated from matches before each fixture date;
  same-day historical matches are excluded to reduce leakage.
- Do not use a match's own final score, shots, possession, cards, xG, or other
  post-kick-off information as explanatory variables.
- Your group should understand and validate the data and calculations, and
  disclose AI assistance according to CDU requirements.

OUTPUT MODEL SCHEMAS
Model 2.1:
- Target: goal_difference
- Eight predictors are listed in collect_data.py and data_quality_report.txt.
Model 2.2:
- Target: goals_scored
- Eight predictors are listed in collect_data.py and data_quality_report.txt.

SOURCES USED BY THE SCRIPT
OpenFootball 2026 World Cup data:
https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.json
International match history:
https://github.com/martj42/international_results
FIFA official statistics:
https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
FIFA fixtures/results:
https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums
April 2026 FIFA ranking report:
https://www.espn.co.uk/football/story/_/id/46664763/fifa-mens-top-50-world-rankings

If a source URL fails or a check fails, do not fabricate missing values. Tell me the error and we can fix the pipeline.
