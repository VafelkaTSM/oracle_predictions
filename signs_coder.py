import pandas as pd
import numpy as np
import json

def coder_no_svd(df):
    df = df.reset_index(drop=True)
    radiant_heroes = pd.DataFrame(df['radiant_heroes'].tolist())
    dire_heroes = pd.DataFrame(df['dire_heroes'].tolist())
    df = df.drop(['radiant_heroes', 'dire_heroes'], axis = 1)
    df = pd.concat([df, radiant_heroes, dire_heroes], axis = 1, ignore_index=True)
    df.columns = (['radiant_team', 'dire_team', 'radiant_streak', 'dire_streak'] + [f'rad_h_{i}' for i in range(5)] + [f'dire_h_{i}' for i in range(5)])
    df['radiant_streak'] = df['radiant_streak'].astype('int64')
    df['dire_streak'] = df['dire_streak'].astype('int64')
    
    return df

def coder_svd_ppmi(df):
    df = df.reset_index(drop=True)
    with open('heroes_list.json', 'r', encoding='utf-8') as f:
        heroes_dict = json.load(f)
    heroes_coder = lambda h: heroes_dict[h]
    radiant_heroes = pd.DataFrame(np.reshape(df['radiant_heroes'].explode().apply(heroes_coder).explode(), 
                                                   (-1, heroes_dict['_length'] * 5))).astype('float64')
    dire_heroes = pd.DataFrame(np.reshape(df['dire_heroes'].explode().apply(heroes_coder).explode(), 
                                                (-1, heroes_dict['_length'] * 5))).astype('float64')
    df = df.drop(['radiant_heroes', 'dire_heroes'], axis = 1)
    df = pd.concat([df, radiant_heroes, dire_heroes], axis = 1)
    df.columns = (['radiant_team', 'dire_team', 'radiant_streak', 'dire_streak'] 
                  + [f'rad_h_{i}' for i in range(heroes_dict['_length'] * 5)] + [f'dire_h_{i}' for i in range(heroes_dict['_length'] * 5)])
    df['radiant_streak'] = df['radiant_streak'].astype('int64')
    df['dire_streak'] = df['dire_streak'].astype('int64')

    return df