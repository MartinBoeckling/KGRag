import json
import pickle
from tqdm.auto import tqdm
import pandas as pd
import sqlite3
import bz2
from sqlitedict import SqliteDict
Movie data
def movie_generate_triples(movies, person_data, year_data):
    triples = []
    movie_id_name = {}
    for movie_key in list(movies.keys()):
        movie_data = movies[movie_key]
        # Basic movie attributes
        movie_number = movie_data.get("id")
        movie_name = movie_data.get("original_title")
        movie_id_name[movie_number] = movie_name
        attributes = [
            ("has_budget", movie_data.get("budget")),
            ("has_id", f'movie_{movie_data.get("id")}'),
            ("has_original_language", movie_data.get("original_language")),
            ("has_original_title", movie_data.get("original_title")),
            ("has_release_date", movie_data.get("release_date")),
            ("has_revenue", movie_data.get("revenue")),
            ("has_title", movie_data.get("title")),
            ("has_rating", movie_data.get("rating")),
            ("has_length", movie_data.get("length")),
        ]
        for attr, value in attributes:
            if value is not None:
                triples.append((movie_data["title"], attr, value))

        # Genres
        for genre in movie_data.get("genres", []):
            triples.append((movie_data["title"], "has_genre", genre["name"]))

        # Cast
        for cast_member in movie_data.get("cast", []):
            triples.append((movie_data["title"], "has_cast", cast_member["name"]))
            triples.append((cast_member["name"], "plays", cast_member["character"]))

        # Crew
        for crew_member in movie_data.get("crew", []):
            triples.append((movie_data["title"], "has_crew", crew_member["name"]))
            triples.append((crew_member["name"], "has_role", crew_member["job"]))

    # Person-Movie Mapping
    for person in person_data:
        for movie_id in person_data.get(person).get("acted_movies", []):
            if movie_id_name.get(movie_id):
                title = movie_id_name[movie_id]
                triples.append((person_data[person].get("name"), "acted_in", title))
        triples.append((person_data[person].get("name"), "has_id", f'person_{person_data[person].get("id")}'))
        triples.append((person_data[person].get("name"), "has_birthday", person_data[person].get("birthday")))

    # year-movie mapping
    for year, data in year_data.items():
        year_movies = data.get("movie_list")
        for movie in year_movies:
            triples.append((f"movie_{movie}", "released_in", year))
        year_oscar_awards = data.get("oscar_awards")
        if year_oscar_awards:
            for year_oscar_data in year_oscar_awards:
                triples.append((year, "edition_of", year_oscar_data["ceremony"]))
                triples.append((year_oscar_data["film"], "nominated_for", year_oscar_data["category"]))
                if year_oscar_data["winner"]:
                    triples.append((year_oscar_data["film"], "won", year_oscar_data["category"]))
                    triples.append((year_oscar_data["name"], "won", year_oscar_data["category"]))
                triples.append((year_oscar_data["name"], "nominated_for", year_oscar_data["category"]))

    return triples

with open("data/CRAG/crag-mock-api-main/cragkg/movie/movie_db.json", "r") as file:
    movie_json = json.load(file)

with open("data/CRAG/crag-mock-api-main/cragkg/movie/person_db.json", "r") as file:
    person_json = json.load(file)

with open("data/CRAG/crag-mock-api-main/cragkg/movie/year_db.json", "r") as file:
    year_movie_json = json.load(file)

movie_triple_data = movie_generate_triples(movie_json, person_json, year_movie_json)

pd_movie_triple_data = pd.DataFrame(movie_triple_data, columns=["subject", "predicate", "object"])

pd_movie_triple_data.to_csv("data/CRAG/triple_data/movie_triple.csv", index=False)

with open("data/CRAG/crag-mock-api-main/cragkg/sports/soccer_team_match_stats.pkl", "rb") as file:
    soccer_data = pickle.load(file)

resetted_soccer_data = soccer_data.reset_index()
resetted_soccer_data["season"] = "23/24"
dict_soccer_data = resetted_soccer_data.to_dict("records")

soccer_triples = []
for row in tqdm(dict_soccer_data):
    # match ID based triples
    match_id = row["game"]
    soccer_triples.append((match_id, "has_home_team", row["team"]))
    soccer_triples.append((match_id, "has_opponent", row["opponent"]))
    soccer_triples.append((match_id, "has_league", row["league"]))
    soccer_triples.append((match_id, "has_season", row["season"]))
    soccer_triples.append((match_id, "played_on", row["date"]))
    soccer_triples.append((match_id, "played_at", row["venue"]))
    soccer_triples.append((match_id, "refereed_by", row["Referee"]))
    home_team_result = row["result"]
    if home_team_result:
        if home_team_result == "W":
            soccer_triples.append((match_id, "won_by", row["team"]))
            soccer_triples.append((match_id, "lost_by", row["opponent"]))
        elif home_team_result == "D":
            soccer_triples.append((match_id, "won_by", row["opponent"]))
            soccer_triples.append((match_id, "lost_by", row["team"]))

    soccer_triples.append((match_id, "had_attendance", row["Attendance"]))
    soccer_triples.append((match_id, f"captain_of_{row["team"]}", row["Captain"]))
    soccer_triples.append((match_id, f"formation_of_{row["team"]}", row["Formation"]))
    soccer_triples.append((match_id, f"posession_of_{row["team"]}", row["Poss"]))
    soccer_triples.append((match_id, f"posession_of_{row["team"]}", row["Poss"]))
    soccer_triples.append((match_id, "week_day", row["day"]))
    soccer_triples.append((match_id, "round_of", row["round"]))
    soccer_triples.append((match_id, "game_time", row["time"]))
    soccer_triples.append((match_id, f"scored_goal_by_{row["team"]}", row["GF"]))
    soccer_triples.append((match_id, f"conceded_goals_by_{row["opponent"]}", row["GA"]))
    soccer_triples.append((match_id, f"scored_goal_by_{row["opponent"]}", row["GA"]))
    soccer_triples.append((match_id, f"conceded_goals_by_{row["team"]}", row["GF"]))
    soccer_triples.append((match_id, f"expected_goals_by_{row["team"]}", row["xG"]))
    soccer_triples.append((match_id, f"expected_goals_by_{row["opponent"]}", row["xGA"]))
    soccer_triples.append((match_id, f"expected_conceded_goals_by_{row["team"]}", row["xGA"]))
    soccer_triples.append((match_id, f"expected_conceded_goals_by_{row["opponent"]}", row["xG"]))
    soccer_triples.append((match_id, "has_notes", row["Notes"]))

pd_soccer_triple_data = pd.DataFrame(soccer_triples, columns=["subject", "predicate", "object"])
pd_soccer_triple_data = pd_soccer_triple_data.dropna()
pd_soccer_triple_data = pd_soccer_triple_data.drop_duplicates()
pd_soccer_triple_data.to_csv("data/CRAG/triple_data/soccer_triple.csv", index=False)

nba_triples = []
nba_conn = sqlite3.connect("data/CRAG/crag-mock-api-main/cragkg/sports/nba.sqlite")
sql_cursor = nba_conn.cursor()
sql_cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
table_overview = sql_cursor.fetchall()
for table in table_overview:
    table_name = table[0]
    print(table_name)
    sql_cursor.execute(f"pragma table_info({table_name})")
    columns = sql_cursor.fetchall()
    column_overview = [column[1] for column in columns]
    sql_cursor.execute(f"SELECT * FROM {table_name}")
    data_aspects = sql_cursor.fetchall()
    pd_data_aspects = pd.DataFrame(data_aspects, columns=column_overview)
    pd_data_aspects.to_csv(f"data/CRAG/pandas_data/{table_name}.csv", index=False)

nba_player = pd.read_csv("data/CRAG/pandas_data/common_player_info.csv")

for _, player_row in nba_player.iterrows():
    nba_triples.append((player_row["person_id"], "has_name", player_row["display_first_last"]))
    nba_triples.append((player_row["person_id"], "born_at", player_row["birthdate"]))
    nba_triples.append((player_row["person_id"], "visited_school", player_row["school"]))
    nba_triples.append((player_row["person_id"], "country", player_row["country"]))
    nba_triples.append((player_row["person_id"], "has_last_affiliation", player_row["last_affiliation"]))
    nba_triples.append((player_row["person_id"], "has_height", player_row["height"]))
    nba_triples.append((player_row["person_id"], "has_weight", player_row["weight"]))
    nba_triples.append((player_row["person_id"], "season_experience", player_row["season_exp"]))
    nba_triples.append((player_row["person_id"], "jersey_number", player_row["season_exp"]))
    nba_triples.append((player_row["person_id"], "nba_position", player_row["position"]))
    nba_triples.append((player_row["person_id"], "rosterstatus", player_row["rosterstatus"]))
    nba_triples.append((player_row["person_id"], "games_played_current_season_flag", player_row["games_played_current_season_flag"]))
    nba_triples.append((player_row["person_id"], "plays_in_team", player_row["team_id"]))
    nba_triples.append((player_row["person_id"], "carrer_time", (str(player_row["from_year"]) + "-" + str(player_row["to_year"]))))
    nba_triples.append((player_row["person_id"], "draft_year", player_row["draft_year"]))
    nba_triples.append((player_row["person_id"], "draft_round", player_row["draft_round"]))
    nba_triples.append((player_row["person_id"], "draft_number", player_row["draft_number"]))
    nba_triples.append((player_row["person_id"], "greatest_75_flag", player_row["greatest_75_flag"]))

nba_games = pd.read_csv("data/CRAG/pandas_data/game.csv")
print(nba_games.columns)
print(nba_games)


Music triples
Artist DB
with open("data/CRAG/crag-mock-api-main/cragkg/music/artist_dict_simplified.pickle", "rb") as file:
    artist_db = pickle.load(file)

music_triples = []
for artist_key, artist_data in artist_db.items():
    music_triples.append((artist_key, "located_in_country", artist_data["country"]))
    music_triples.append((artist_key, "birth_date", artist_data["birth_date"]))
    music_triples.append((artist_key, "has_members", ', '.join(artist_data["members"])))
    music_triples.append((artist_key, "end_date", artist_data["end_date"]))

# Artist Work DB
with open("data/CRAG/crag-mock-api-main/cragkg/music/artist_work_dict.pickle", "rb") as file:
    artist_work_db = pickle.load(file)


for artist_work_key, artist_work_data in artist_work_db.items():
    for artist_work in artist_work_data:
        music_triples.append((artist_work_key, "has_artist_work", artist_work))


# Grammy Data
grammy_db = pd.read_pickle("data/CRAG/crag-mock-api-main/cragkg/music/grammy_df.pickle")
for _, grammy_row in grammy_db.iterrows():
    grammy_event = grammy_row["title"]
    music_triples.append((grammy_event, "in_year_of", grammy_row["year"]))
    music_triples.append((grammy_event, f"winner_{grammy_row["category"]}", grammy_row["artist"]))
    music_triples.append((grammy_row["artist"], "winner_song", grammy_row["nominee"]))
    music_triples.append((grammy_row["nominee"], "has_workers", grammy_row["workers"]))

# Billboard Ranking
with open("data/CRAG/crag-mock-api-main/cragkg/music/rank_dict_hot100.pickle", "rb") as file:
    music_rank_100_db = pickle.load(file)

for music_rank_key, music_rank_list in music_rank_100_db.items():
    for music_rank_value in music_rank_list:
        music_triples.append((f"Hot_100_Billboard_Rank_{music_rank_key}", "has_artist", music_rank_value["Artist"]))
        music_triples.append((f"Hot_100_Billboard_Rank_{music_rank_key}", "has_song", music_rank_value["Song"]))
        music_triples.append((music_rank_value["Song"], f"Hot_100_Billboard_Rank_{music_rank_key}_Rank_Date", music_rank_value["Date"]))

# Music Song Ranking Position
with open("data/CRAG/crag-mock-api-main/cragkg/music/song_dict_hot100.pickle", "rb") as file:
    music_song_100_db = pickle.load(file)


for music_song_key, music_rank_list in music_song_100_db.items():
    for song_date, song_ranking_list in music_rank_list.items():
        music_triples.append((music_song_key, "billboard_ranking_date", song_ranking_list[0]))
        music_triples.append((music_song_key, "billboard_ranking_artist", song_ranking_list[2]))
        music_triples.append((music_song_key, "rank_last_week", song_ranking_list[3]))
        music_triples.append((music_song_key, "weeks_in_chart", song_ranking_list[4]))
        music_triples.append((music_song_key, "top_position", song_ranking_list[5]))
        music_triples.append((music_song_key, "rank", song_ranking_list[6]))

with open("data/CRAG/crag-mock-api-main/cragkg/music/song_dict_simplified.pickle", "rb") as file:
    song_simplified_db = pickle.load(file)

for song_sim_key, song_sim_item in song_simplified_db.items():
    music_triples.append((song_sim_key, "artist", song_sim_item["author"]))
    music_triples.append((song_sim_key, "country", song_sim_item["country"]))
    music_triples.append((song_sim_key, "released_in", song_sim_item["date"]))

music_triple_data = pd.DataFrame(music_triples, columns=["subject", "predicate", "object"])

music_triple_data = music_triple_data.drop_duplicates()
music_triple_data.to_csv("data/CRAG/triple_data/musc_triple.csv", index=False)

# Open KG
open_kg = {}
open_triples = []
with bz2.open("data/CRAG/crag-mock-api-main/cragkg/open/kg.0.jsonl.bz2", "rt", encoding='utf8') as f:
    l = f.readline()
    while l:
        l = json.loads(l)
        open_kg[l[0]] = l[1]
        l = f.readline()

with bz2.open("data/CRAG/crag-mock-api-main/cragkg/open/kg.1.jsonl.bz2", "rt", encoding='utf8') as f:
    l = f.readline()
    while l:
        l = json.loads(l)
        open_kg[l[0]] = l[1]
        l = f.readline()

for key, value in open_kg.items():
    open_triples.append((key, "has_summary", value["summary_text"]))

open_kg_df = pd.DataFrame(open_triples, columns=["subject", "predicate", "object"])
open_kg_df.to_csv("data/CRAG/triple_data/open_kg.csv", index=False)

Finance
finance_triples = []
# Company-Symbol
company_names = pd.read_csv("data/CRAG/crag-mock-api-main/cragkg/finance/company_name.dict")
for _, company_row in company_names.iterrows():
    finance_triples.append((company_row["Name"], "has_ticker_name", company_row["Symbol"]))

finance_detailed_price_db = SqliteDict("data/CRAG/crag-mock-api-main/cragkg/finance/finance_detailed_price.sqlite")
for key, item in finance_detailed_price_db.items():
    for trade_date, trade_parameters in item.items():
        finance_triples.append((f"{key}", "traded_at", f"{key}_{trade_date}"))
        finance_triples.append((f"{key}_{trade_date}", "opening_price", trade_parameters["Open"]))
        finance_triples.append((f"{key}_{trade_date}", "high_price", trade_parameters["High"]))
        finance_triples.append((f"{key}_{trade_date}", "low_price", trade_parameters["Low"]))
        finance_triples.append((f"{key}_{trade_date}", "closing_price", trade_parameters["Close"]))
        finance_triples.append((f"{key}_{trade_date}", "trade_volume", trade_parameters["Volume"]))

finance_dividend_db = SqliteDict("data/CRAG/crag-mock-api-main/cragkg/finance/finance_dividend.sqlite")

for key, item in finance_dividend_db.items():
    if item:
        finance_triples.append((key, "has_dividend_history", str(item)))
    else:
        finance_triples.append((key, "has_dividend_history", "None"))

finance_info_db = SqliteDict("data/CRAG/crag-mock-api-main/cragkg/finance/finance_info.sqlite")
for key, item in finance_info_db.items():
    company_country = item.pop("country")
    company_city = item.pop("city")
    company_industry = item.pop("industry")
    company_sector = item.pop("sectorKey")
    company_summary = item.pop("longBusinessSummary")
    company_officers = item.pop("companyOfficers")
    finance_triples.append((key, "location_country", company_country))
    finance_triples.append((key, "location_city", company_city))
    finance_triples.append((key, "industry", company_industry))
    finance_triples.append((key, "sector", company_sector))
    finance_triples.append((key, "company_summary", company_sector))
    finance_triples.append((key, "company_officers", str(company_officers)))
    finance_triples.append((key, "company_information", str(item)))
    break

finance_marketcap_db = SqliteDict("data/CRAG/crag-mock-api-main/cragkg/finance/finance_marketcap.sqlite")

for key, item in finance_marketcap_db.items():
    finance_triples.append((key, "has_market_cap", f"{item} USD"))

finance_price_db = SqliteDict("data/CRAG/crag-mock-api-main/cragkg/finance/finance_price.sqlite")
for key, item in finance_price_db.items():
    for trade_date, trade_parameters in item.items():
        finance_triples.append((f"{key}", "traded_at", f"{key}_{trade_date}"))
        finance_triples.append((f"{key}_{trade_date}", "opening_price", trade_parameters["Open"]))
        finance_triples.append((f"{key}_{trade_date}", "high_price", trade_parameters["High"]))
        finance_triples.append((f"{key}_{trade_date}", "low_price", trade_parameters["Low"]))
        finance_triples.append((f"{key}_{trade_date}", "closing_price", trade_parameters["Close"]))
        finance_triples.append((f"{key}_{trade_date}", "trade_volume", trade_parameters["Volume"]))


finance_triple_data = pd.DataFrame(finance_triples, columns=["subject", "predicate", "object"])
finance_triple_data = finance_triple_data.astype("str")
finance_triple_data = finance_triple_data.drop_duplicates()
finance_triple_data.to_csv("data/CRAG/triple_data/finance_triple.csv", index=False)
