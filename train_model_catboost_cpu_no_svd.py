import pandas as pd
import sys
from scipy.sparse import csr_matrix, diags
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import json

np.random.seed(42)
separator_1 = 1000
separator_2 = 1000

try:
    path_to_table_1 = sys.argv[1]
    path_to_table_2 = sys.argv[2]
except:
    raise Exception('The first argument must be the path to the main table, and the second—the path to the table for fine-tuning.')

df_1 = pd.read_parquet(path_to_table_1, 
                       columns = ['radiant_team', 'dire_team', 'radiant_streak', 'dire_streak', 'radiant_heroes', 'dire_heroes', 'radiant_win'], 
                       engine = 'pyarrow')
df_2 = pd.read_parquet(path_to_table_2, 
                       columns = ['radiant_team', 'dire_team', 'radiant_streak', 'dire_streak', 'radiant_heroes', 'dire_heroes', 'radiant_win'],
                       engine = 'pyarrow')

# We clear the table of rare teams.
table_count= pd.concat([df_1['radiant_team'], df_2['radiant_team'],
                         df_1['dire_team'], df_2['dire_team']], 
                         ignore_index = True)
table_count = table_count.value_counts()
rare_teams = table_count[table_count < 10].index.to_list()
df_1 = df_1[~df_1['radiant_team'].isin(rare_teams)]
df_1 = df_1[~df_1['dire_team'].isin(rare_teams)]
df_1 = df_1.reset_index(drop=True)
df_2 = df_2[~df_2['radiant_team'].isin(rare_teams)]
df_2 = df_2[~df_2['dire_team'].isin(rare_teams)]
df_2 = shuffle(df_2, random_state=42)
df_2 = df_2.reset_index(drop=True)
df_2 = df_2.iloc[:len(df_2) - separator_2]

# Training CatBoost boosting with automatic hyperparameter tuning using Optuna.
def sep_heroes(df):
    df_radiant = pd.DataFrame(df['radiant_heroes'].tolist())
    df_dire = pd.DataFrame(df['dire_heroes'].tolist())
    df = df.drop(['radiant_heroes', 'dire_heroes'], axis = 1)
    df = pd.concat([df, df_radiant, df_dire], axis = 1, ignore_index=True)
    hero_columns_name = (['radiant_team', 'dire_team', 'radiant_streak', 'dire_streak']
                         + [f'rad_h_{i}' for i in range(5)] + [f'dire_h_{i}' for i in range(5)])
    df.columns = hero_columns_name

    return df

df_1['radiant_streak'] = df_1['radiant_streak'].astype('int64')
df_1['dire_streak'] = df_1['dire_streak'].astype('int64')
df_2['radiant_streak'] = df_2['radiant_streak'].astype('int64')
df_2['dire_streak'] = df_2['dire_streak'].astype('int64')
X_1 = df_1.drop(['radiant_win'], axis = 1)
X_1 = sep_heroes(X_1)
Y_1 = df_1['radiant_win']
X_2 = df_2.drop(['radiant_win'], axis = 1)
X_2 = sep_heroes(X_2)
Y_2 = df_2['radiant_win']
X_1, Y_1 = shuffle(X_1, Y_1, random_state=42)
X_2, Y_2 = shuffle(X_2, Y_2, random_state=42)
X_1, Y_1 = X_1.reset_index(drop=True), Y_1.reset_index(drop=True)
X_2, Y_2 = X_2.reset_index(drop=True), Y_2.reset_index(drop=True)
X_2_val_end, Y_2_val_end = X_2.iloc[len(X_2) - separator_1:], Y_2.iloc[len(X_2) - separator_1:]
X_2, Y_2 = X_2.iloc[:len(X_2) - separator_1], Y_2.iloc[:len(X_2) - separator_1]
col_names = ['radiant_team', 'dire_team'] + [f'rad_h_{i}' for i in range(5)] + [f'dire_h_{i}' for i in range(5)]
    
def objective(trial):
    params_1 = {
        'objective': 'Logloss',
        'cat_features': col_names,
        'thread_count': -1,
        'early_stopping_rounds': 20,
        'use_best_model': True,
        'random_seed': 42,
        'iterations': 1000,
        'depth': trial.suggest_int('depth_1', 1, 8),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel_1', 0.01, 1),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type_1', ['Bayesian', 'Bernoulli', 'MVS']),
        'learning_rate': trial.suggest_float('learning_rate_1', 1e-3, 0.4, log = True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg_1', 1e-2, 10.0, log = True),
        'random_strength': trial.suggest_float('random_strength_1', 1e-8, 10.0, log = True)
    }
    params_2 = {
        'objective': 'Logloss',
        'cat_features': col_names,
        'thread_count': -1,
        'early_stopping_rounds': 20,
        'use_best_model': True,
        'random_seed': 42,
        'iterations': 500,
        'depth': trial.suggest_int('depth_2', 1, 8),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel_2', 0.01, 1),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type_2', ['Bayesian', 'Bernoulli', 'MVS']),
        'learning_rate': trial.suggest_float('learning_rate_2', 1e-3, 0.4, log = True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg_2', 1e-2, 10.0, log = True),
        'random_strength': trial.suggest_float('random_strength_2', 1e-8, 10.0, log = True)
    }

    if params_1['bootstrap_type'] == 'Bayesian':
        params_1['bagging_temperature'] = trial.suggest_float('bagging_temperature_1', 0, 10)
    elif params_1['bootstrap_type'] == 'Bernoulli':
        params_1['subsample'] = trial.suggest_float('subsample_1', 0.1, 1)
    if params_2['bootstrap_type'] == 'Bayesian':
        params_2['bagging_temperature'] = trial.suggest_float('bagging_temperature_2', 0, 10)
    elif params_2['bootstrap_type'] == 'Bernoulli':
        params_2['subsample'] = trial.suggest_float('subsample_2', 0.1, 1)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_2):
        X_2_tr, Y_2_tr = X_2.iloc[train_idx], Y_2.iloc[train_idx]
        X_2_val, Y_2_val = X_2.iloc[val_idx], Y_2.iloc[val_idx]

        model_1 = CatBoostClassifier(**params_1, silent = True)
        model_1.fit(X_1, Y_1, eval_set =(X_2_val, Y_2_val))

        model_2 = CatBoostClassifier(**params_2, silent = True)
        model_2.fit(X_2_tr, Y_2_tr, eval_set =(X_2_val, Y_2_val), init_model = model_1)

        preds = model_2.predict_proba(X_2_val)
        scores.append(log_loss(Y_2_val, preds))
    score = sum(scores) / len(scores)

    return score

sampler = optuna.samplers.TPESampler(seed=42)
study_catboost = optuna.create_study(direction='minimize', sampler=sampler)
study_catboost.optimize(objective, n_trials = 200, show_progress_bar = True)
params = study_catboost.best_params
other_params = {'objective': 'Logloss',
                'cat_features': col_names,
                'random_seed': 42,
                'early_stopping_rounds': 20,
                'use_best_model': True,
                'boosting_type': 'Ordered'}
params_1 = {k[:-2]: params[k] for k in params.keys() if '_1' in k}
params_1.update(other_params)
params_2 = {k[:-2]: params[k] for k in params.keys() if '_2' in k}
params_2.update(other_params)

model_1 = CatBoostClassifier(**params_1)
model_1.fit(X_1, Y_1, eval_set=(X_2_val_end, Y_2_val_end))
model_2 = CatBoostClassifier(**params_2)
model_2.fit(X_2, Y_2, eval_set=(X_2_val_end, Y_2_val_end), init_model = model_1)

# Saving the model.
model_2.save_model('model_catboost_cpu_no_svd.json')
print(model_1.get_params(), model_2.get_params())
with open('heroes_list.json', 'w', encoding='utf-8') as f:
    heroes_list = pd.concat([df_1['radiant_heroes'].explode(), df_1['dire_heroes'].explode(), 
                             df_2['radiant_heroes'].explode(), df_2['dire_heroes'].explode()], 
                             axis=0, ignore_index=True).unique().tolist()
    json.dump(heroes_list, f, ensure_ascii = False)
with open('teams_list.json', 'w', encoding='utf-8') as f:
    teams_list = pd.concat([df_1['radiant_team'], df_1['dire_team'], 
                            df_2['radiant_team'], df_2['dire_team']], 
                            axis=0, ignore_index=True).unique().tolist()
    json.dump(teams_list, f, ensure_ascii = False)

joblib.dump(study_catboost, 'history_study_catboost_cpu_no_svd.pkl')
plot_optimization_history(study_catboost).show()
plot_param_importances(study_catboost).show()