

from pathlib import Path
import json
import re
import unicodedata
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw"
RAW.mkdir(exist_ok=True)

WORLD_CUP_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
HISTORY_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Rankings as reported for 1 April 2026. This is a pre-tournament snapshot.
# Source: ESPN's report of the April 1, 2026 FIFA Men's Ranking.
RANKS = {
    "France": 1, "Spain": 2, "Argentina": 3, "England": 4, "Portugal": 5,
    "Brazil": 6, "Netherlands": 7, "Morocco": 8, "Belgium": 9, "Germany": 10,
    "Croatia": 11, "Italy": 12, "Colombia": 13, "Senegal": 14, "Mexico": 15,
    "USA": 16, "Uruguay": 17, "Japan": 18, "Switzerland": 19, "Denmark": 20,
    "Iran": 21, "Turkey": 22, "Ecuador": 23, "Austria": 24,
    "South Korea": 25, "Nigeria": 26, "Australia": 27, "Algeria": 28,
    "Egypt": 29, "Canada": 30, "Norway": 31, "Ukraine": 32, "Panama": 33,
    "Ivory Coast": 34, "Poland": 35, "Russia": 36, "Wales": 37, "Sweden": 38,
    "Serbia": 39, "Paraguay": 40, "Czech Republic": 41, "Hungary": 42,
    "Scotland": 43, "Tunisia": 44, "Cameroon": 45, "DR Congo": 46,
    "Greece": 47, "Slovakia": 48, "Venezuela": 49, "Uzbekistan": 50,
    "Qatar": 55, "Iraq": 57, "South Africa": 60, "Saudi Arabia": 61,
    "Jordan": 63, "Bosnia and Herzegovina": 65, "Cape Verde": 69,
    "Ghana": 74, "Curaçao": 82, "Haiti": 84, "New Zealand": 85,
}

ALIASES = {
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "curacao": "Curaçao",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cote d’ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "côte d’ivoire": "Ivory Coast",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "cape verde islands": "Cape Verde",
    "cape verde": "Cape Verde",
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "ir iran": "Iran",
    "iran": "Iran",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "new zealand": "New Zealand",
    "mexico": "Mexico",
}


def norm_key(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().strip()
    value = re.sub(r"[’']", "'", value)
    value = re.sub(r"[^a-z0-9&' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_team(value):
    key = norm_key(value)
    # Alias keys are normalized too, so accent/case variants resolve consistently.
    aliases = {norm_key(k): v for k, v in ALIASES.items()}
    return aliases.get(key, str(value).strip())


def download(url, destination):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def read_world_cup_matches(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, match in enumerate(payload["matches"], start=1):
        score = match.get("score") or {}
        ft = score.get("ft")
        if not ft or len(ft) != 2:
            raise ValueError(f"Missing full-time score for match {i}: {match}")
        # In this source, "et" records extra-time goals separately from regulation.
        et = score.get("et", [0, 0]) or [0, 0]
        goals_a = int(ft[0]) + int(et[0])
        goals_b = int(ft[1]) + int(et[1])
        rows.append({
            "match_id": match.get("num", i),
            "date": match["date"],
            "round": match.get("round", ""),
            "team_a": canonical_team(match["team1"]),
            "team_b": canonical_team(match["team2"]),
            "team_a_goals": goals_a,
            "team_b_goals": goals_b,
            "goal_difference": goals_a - goals_b,
            "venue": match.get("ground", ""),
            "source_team_a_name": match["team1"],
            "source_team_b_name": match["team2"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["date", "match_id"]).reset_index(drop=True)
    if len(df) != 104:
        raise ValueError(f"Expected 104 World Cup matches, found {len(df)}.")
    return df


def make_team_history(history_df):
    h = history_df.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["home_score"] = pd.to_numeric(h["home_score"], errors="coerce")
    h["away_score"] = pd.to_numeric(h["away_score"], errors="coerce")
    h = h.dropna(subset=["date", "home_score", "away_score"])
    h["home_team"] = h["home_team"].map(canonical_team)
    h["away_team"] = h["away_team"].map(canonical_team)

    home = pd.DataFrame({
        "date": h["date"], "team": h["home_team"], "opponent": h["away_team"],
        "goals_for": h["home_score"], "goals_against": h["away_score"],
    })
    away = pd.DataFrame({
        "date": h["date"], "team": h["away_team"], "opponent": h["home_team"],
        "goals_for": h["away_score"], "goals_against": h["home_score"],
    })
    out = pd.concat([home, away], ignore_index=True)
    out["goal_difference"] = out["goals_for"] - out["goals_against"]
    out["win"] = (out["goals_for"] > out["goals_against"]).astype(int)
    out["loss"] = (out["goals_for"] < out["goals_against"]).astype(int)
    out["clean_sheet"] = (out["goals_against"] == 0).astype(int)
    return out.sort_values(["date", "team"]).reset_index(drop=True)


def recent_stats(team_history, team, fixture_date):
    # Strictly earlier calendar dates only as this avoids same-day time leakage.
    prior = team_history[
        (team_history["team"] == team) &
        (team_history["date"] < pd.Timestamp(fixture_date))
    ].sort_values("date")
    last5 = prior.tail(5)
    if last5.empty:
        return {
            "n_history": 0, "win_rate": None, "avg_scored": None,
            "avg_conceded": None, "avg_goal_difference": None,
            "last_match_goals_scored": None, "clean_sheet_rate": None,
            "loss_rate": None,
        }
    return {
        "n_history": int(len(last5)),
        "win_rate": float(last5["win"].mean()),
        "avg_scored": float(last5["goals_for"].mean()),
        "avg_conceded": float(last5["goals_against"].mean()),
        "avg_goal_difference": float(last5["goal_difference"].mean()),
        "last_match_goals_scored": float(prior.iloc[-1]["goals_for"]),
        "clean_sheet_rate": float(last5["clean_sheet"].mean()),
        "loss_rate": float(last5["loss"].mean()),
    }


def main():
    wc_path = download(WORLD_CUP_URL, RAW / "worldcup_2026_source.json")
    hist_path = download(HISTORY_URL, RAW / "international_results_source.csv")

    matches = read_world_cup_matches(wc_path)
    matches.to_csv(ROOT / "world_cup_2026_matches.csv", index=False)

    hist = pd.read_csv(hist_path)
    team_history = make_team_history(hist)

    # Store the pre-tournament ranking lookup for transparent reproducibility.
    rank_df = pd.DataFrame(
        [{"team": team, "fifa_rank_2026_04_01": rank}
            for team, rank in sorted(RANKS.items(), key=lambda x: x[1])]
    )
    rank_df.to_csv(RAW / "fifa_rankings_april_2026.csv", index=False)

    rows21, rows22 = [], []
    rank_missing = set()
    history_missing = []

    for _, m in matches.iterrows():
        d = m["date"]
        a, b = m["team_a"], m["team_b"]
        if a not in RANKS:
            rank_missing.add(a)
        if b not in RANKS:
            rank_missing.add(b)
        sa, sb = recent_stats(team_history, a, d), recent_stats(
            team_history, b, d)
        if sa["n_history"] == 0:
            history_missing.append((d, a))
        if sb["n_history"] == 0:
            history_missing.append((d, b))

        rows21.append({
            "match_id": m["match_id"], "date": d, "team_a": a, "team_b": b,
            "team_a_goals": m["team_a_goals"], "team_b_goals": m["team_b_goals"],
            "goal_difference": m["goal_difference"],
            "team_a_fifa_rank": RANKS.get(a),
            "team_b_fifa_rank": RANKS.get(b),
            "team_a_win_rate_last5": sa["win_rate"],
            "team_b_win_rate_last5": sb["win_rate"],
            "team_a_avg_goals_scored_last5": sa["avg_scored"],
            "team_b_avg_goals_scored_last5": sb["avg_scored"],
            "team_a_avg_goals_conceded_last5": sa["avg_conceded"],
            "team_b_avg_goals_conceded_last5": sb["avg_conceded"],
            "team_a_history_matches_used": sa["n_history"],
            "team_b_history_matches_used": sb["n_history"],
        })

        rows22.append({
            "match_id": m["match_id"], "date": d, "team": a, "opponent": b,
            "goals_scored": m["team_a_goals"], "goals_conceded": m["team_b_goals"],
            "team_avg_goal_difference_last5": sa["avg_goal_difference"],
            "opponent_avg_goal_difference_last5": sb["avg_goal_difference"],
            "team_last_match_goals_scored": sa["last_match_goals_scored"],
            "opponent_last_match_goals_scored": sb["last_match_goals_scored"],
            "team_clean_sheet_rate_last5": sa["clean_sheet_rate"],
            "opponent_clean_sheet_rate_last5": sb["clean_sheet_rate"],
            "team_loss_rate_last5": sa["loss_rate"],
            "opponent_loss_rate_last5": sb["loss_rate"],
            "team_history_matches_used": sa["n_history"],
            "opponent_history_matches_used": sb["n_history"],
        })
        rows22.append({
            "match_id": m["match_id"], "date": d, "team": b, "opponent": a,
            "goals_scored": m["team_b_goals"], "goals_conceded": m["team_a_goals"],
            "team_avg_goal_difference_last5": sb["avg_goal_difference"],
            "opponent_avg_goal_difference_last5": sa["avg_goal_difference"],
            "team_last_match_goals_scored": sb["last_match_goals_scored"],
            "opponent_last_match_goals_scored": sa["last_match_goals_scored"],
            "team_clean_sheet_rate_last5": sb["clean_sheet_rate"],
            "opponent_clean_sheet_rate_last5": sa["clean_sheet_rate"],
            "team_loss_rate_last5": sb["loss_rate"],
            "opponent_loss_rate_last5": sa["loss_rate"],
            "team_history_matches_used": sb["n_history"],
            "opponent_history_matches_used": sa["n_history"],
        })

    m21 = pd.DataFrame(rows21)
    m22 = pd.DataFrame(rows22)
    m21.to_csv(ROOT / "model_2_1_goal_difference.csv", index=False)
    m22.to_csv(ROOT / "model_2_2_goals_scored.csv", index=False)

    # Quality checks: expected row counts, feature counts, and missing values.
    features21 = [
        "team_a_fifa_rank", "team_b_fifa_rank",
        "team_a_win_rate_last5", "team_b_win_rate_last5",
        "team_a_avg_goals_scored_last5", "team_b_avg_goals_scored_last5",
        "team_a_avg_goals_conceded_last5", "team_b_avg_goals_conceded_last5",
    ]
    features22 = [
        "team_avg_goal_difference_last5", "opponent_avg_goal_difference_last5",
        "team_last_match_goals_scored", "opponent_last_match_goals_scored",
        "team_clean_sheet_rate_last5", "opponent_clean_sheet_rate_last5",
        "team_loss_rate_last5", "opponent_loss_rate_last5",
    ]

    lines = [
        "HIT140 Assignment 3 — data quality report",
        "==========================================",
        f"World Cup match rows: {len(matches)} (expected 104)",
        f"Model 2.1 rows: {len(m21)} (expected 104)",
        f"Model 2.1 explanatory variables: {len(features21)} (expected 8)",
        f"Model 2.2 rows: {len(m22)} (expected 208)",
        f"Model 2.2 explanatory variables: {len(features22)} (expected 8)",
        f"Model 2.1 missing predictor cells: {int(m21[features21].isna().sum().sum())}",
        f"Model 2.2 missing predictor cells: {int(m22[features22].isna().sum().sum())}",
        f"Teams missing a ranking: {sorted(rank_missing)}",
        f"Team-fixture combinations with no prior history: {history_missing[:20]}",
        "",
        "Feature definitions:",
        "- Last-five rates/averages use the most recent five international matches",
        "  with a calendar date strictly before the World Cup fixture date.",
        "- If fewer than five earlier matches exist, available matches are used;",
        "  inspect the *_history_matches_used columns.",
        "- FIFA rankings are the 1 April 2026 pre-tournament snapshot.",
        "- Model 2.1 target is goal_difference = team A goals - team B goals.",
        "- Model 2.2 has two rows per match; target is goals_scored by that team.",
        "",
        "Source links:",
        "FIFA official statistics: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics",
        "FIFA fixtures/results: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums",
        "The Stats Don't Lie: https://www.thestatsdontlie.com/football/world-cup-2026/",
        "FBref: https://fbref.com/en/",
        "Structured 2026 match data (OpenFootball): " + WORLD_CUP_URL,
        "International results history (Mart Jürisoo): " + HISTORY_URL,
        "April 2026 ranking report (ESPN): https://www.espn.co.uk/football/story/_/id/46664763/fifa-mens-top-50-world-rankings",
        "",
        "Caution: review all missing values and name mappings before modelling.",
        "Do not use same-match shots, possession, cards, or other in-game stats as predictors.",
    ]
    (ROOT / "data_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Data collection complete.")
    print(
        f"Matches: {len(matches)} | Model 2.1: {len(m21)} rows | Model 2.2: {len(m22)} rows")
    print(
        f"Model 2.1 missing predictor cells: {int(m21[features21].isna().sum().sum())}")
    print(
        f"Model 2.2 missing predictor cells: {int(m22[features22].isna().sum().sum())}")
    print("Read data_quality_report.txt and inspect the *_history_matches_used columns.")


if __name__ == "__main__":
    main()
