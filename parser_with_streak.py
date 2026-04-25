import httpx
import pandas as pd
import sys
import time

def get_data_from_explorer(sql_query):
    url = 'https://api.opendota.com/api/explorer'

    with httpx.Client(http2=True, timeout=60) as client:
        i = 0
        j = 0
        while True:
            try:
                response = client.get(url, params={'sql': sql_query})
            except httpx.TimeoutException:
                print(f'The timeout has expired. Retrying in {60} seconds. Number of timeout delays for this request: {i}.')
                time.sleep(60)
                i += 1
                continue
            if response.status_code == 200:
                return response.json().get('rows', [])
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    delay = int(retry_after)  
                else:
                    delay = 60
                j += 1
                print(f'Error 429: Too Many Requests. Retrying in {delay} seconds. Number of delays due to Error 429 for this request: {j}.')
                time.sleep(delay)
            else:
                response.raise_for_status()

def get_matches_streak(start_date, end_date):
    query = f'''
    WITH series_info_1 AS (
        SELECT 
            series_id,
            CASE WHEN radiant_team_id < dire_team_id THEN radiant_team_id ELSE dire_team_id END as t1,
            CASE WHEN radiant_team_id > dire_team_id THEN radiant_team_id ELSE dire_team_id END as t2, 
            start_time,
            CASE WHEN (radiant_win AND radiant_team_id < dire_team_id) 
                    OR (NOT radiant_win AND radiant_team_id > dire_team_id) THEN 1 ELSE 0 END as t1_win,
            CASE WHEN (radiant_win AND radiant_team_id > dire_team_id) 
                    OR (NOT radiant_win AND radiant_team_id < dire_team_id) THEN 1 ELSE 0 END as t2_win
        FROM matches
        WHERE series_id IS NOT NULL AND series_id > 0
        AND start_time >= {start_date:.0f} - 864000
    ),
    series_info_2 AS (
        SELECT
            series_id,
            t1, 
            t2, 
            MAX(start_time) as series_end_time,
            SUM(t1_win) as t1_wins,
            SUM(t2_win) as t2_wins
        FROM series_info_1
        GROUP BY series_id, t1, t2
    ),
    series_winners AS (
        SELECT 
            series_id, series_end_time,
            t1 as team_id, (t1_wins > t2_wins) as won FROM series_info_2
        UNION ALL
        SELECT 
            series_id, series_end_time,
            t2 as team_id, (t2_wins > t1_wins) as won FROM series_info_2
    ),
    streak_groups AS (
        SELECT 
            team_id, series_id, series_end_time, won,
            ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY series_end_time) - 
            ROW_NUMBER() OVER (PARTITION BY team_id, won ORDER BY series_end_time) as group_id
        FROM series_winners
    ),
    streak_counters AS (
        SELECT 
            team_id, series_id, series_end_time, won,
            ROW_NUMBER() OVER (PARTITION BY team_id, won, group_id ORDER BY series_end_time) as current_streak_val
        FROM streak_groups
    ),
    calculated_series_streaks AS (
        SELECT
            team_id,
            series_id,
            COALESCE(LAG(CASE WHEN won THEN '+' ELSE '-' END || current_streak_val) 
                OVER (PARTITION BY team_id ORDER BY series_end_time), '0') as series_streak_before
        FROM streak_counters
    )

    SELECT
        m.match_id,
        rss.series_streak_before as radiant_streak,
        dss.series_streak_before as dire_streak
    FROM matches m
    LEFT JOIN teams t1 ON m.radiant_team_id = t1.team_id
    LEFT JOIN teams t2 ON m.dire_team_id = t2.team_id
    LEFT JOIN calculated_series_streaks rss ON m.series_id = rss.series_id AND m.radiant_team_id = rss.team_id
    LEFT JOIN calculated_series_streaks dss ON m.series_id = dss.series_id AND m.dire_team_id = dss.team_id
    WHERE m.start_time >= {start_date:.0f} 
    AND m.start_time <= {end_date:.0f}
    AND t1.name IS NOT NULL 
    AND t2.name IS NOT NULL
    AND m.series_id IS NOT NULL
    ORDER BY m.start_time DESC
    '''

    return get_data_from_explorer(query)

def get_matches_heroes(start_date, end_date):
    query = f'''
    SELECT
    m.match_id,
    t1.name as radiant_team,
    t2.name as dire_team,
    m.start_time,
    
    json_agg(
        h.localized_name 
        ORDER BY pm.lane_role ASC, pm.last_hits DESC
    ) FILTER (WHERE pm.player_slot < 128) as radiant_heroes,
    json_agg(
        h.localized_name 
        ORDER BY pm.lane_role ASC, pm.last_hits DESC
    ) FILTER (WHERE pm.player_slot >= 128) as dire_heroes,
    m.radiant_win

    FROM matches m
    JOIN player_matches pm ON m.match_id = pm.match_id
    JOIN heroes h ON pm.hero_id = h.id -- Присоединяем таблицу героев
    LEFT JOIN teams t1 ON m.radiant_team_id = t1.team_id
    LEFT JOIN teams t2 ON m.dire_team_id = t2.team_id
    JOIN leagues l ON m.leagueid = l.leagueid
    WHERE m.start_time >= {start_date:.0f}
    AND m.start_time <= {end_date:.0f}
    AND l.tier IN ('premium', 'professional')
    AND t1.name IS NOT NULL
    AND t2.name IS NOT NULL
    GROUP BY m.match_id, t1.name, t2.name, m.start_time, m.radiant_win
    ORDER BY m.start_time DESC
    '''

    return get_data_from_explorer(query)

def get_date():
    try:
        calendar_start_date = sys.argv[1]
        calendar_end_date = sys.argv[2]
    except:
        raise Exception('Not all dates have been provided in the program arguments.')
    
    start_date = pd.to_datetime(f'{calendar_start_date} 00:00:00').timestamp()
    end_date = pd.to_datetime(f'{calendar_end_date} 00:00:00').timestamp()
    if start_date > end_date:
        raise Exception('The first date must be earlier than the first date.')

    return start_date, end_date, calendar_start_date, calendar_end_date

start_date, end_date, calendar_start_date, calendar_end_date = get_date()
matches = pd.DataFrame(columns = ['match_id', 'radiant_team', 'dire_team', 'start_time', 
                                  'radiant_heroes', 'dire_heroes', 'radiant_win', 'radiant_streak', 'dire_streak'])
progress_date = start_date
while progress_date < end_date:
    new_matches_heroes = pd.DataFrame(get_matches_heroes(progress_date, progress_date + 86400))
    new_matches_streak = pd.DataFrame(get_matches_streak(progress_date, progress_date + 86400))
    new_matches = pd.merge(new_matches_heroes, new_matches_streak, on = 'match_id', how = 'inner')
    matches = pd.concat([matches, new_matches], ignore_index = True, axis = 0)
    progress_date += 86400
    print(f'Прогресс: {((progress_date - start_date) / 86400):.0f} / {((end_date - start_date) / 86400):.0f}')
    
matches = matches.sort_values('start_time', ascending=False)
matches = matches.dropna()
matches = matches.reset_index()
matches.to_parquet(f'pro_matches_from_{calendar_start_date}_to_{calendar_end_date}.parquet', engine='pyarrow')