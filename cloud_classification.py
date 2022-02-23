import pandas as pd
import xgboost as xgb
import os
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder)
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     StratifiedKFold)
from sklearn.metrics import (confusion_matrix,
                             plot_confusion_matrix,
                             log_loss)
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import ast
import time
import joblib

from nsrdb.utilities.statistics import mae_perc, mbe_perc

working_dir = '/projects/pxs/mlclouds/cloud_id/bnb/data'
output_dir = '/projects/pxs/mlclouds/cloud_id/bnb/output'
batch_model_info_file = f'{output_dir}/batch_model_info.csv'

cloud_map = {'clearsky': 0, 'water': 1, 'ice': 2}

features = [
    'solar_zenith_angle',
    'refl_0_65um_nom',
    'refl_0_65um_nom_stddev_3x3',
    'refl_3_75um_nom',
    'temp_3_75um_nom',
    'temp_11_0um_nom',
    'temp_11_0um_nom_stddev_3x3',
    'cloud_probability',
    'cloud_fraction',
    'air_temperature',
    'dew_point',
    'relative_humidity',
    'total_precipitable_water',
    'surface_albedo',
    'alpha',
    'aod',
    'ozone',
    'ssa',
    'surface_pressure',
    'cld_opd_mlclouds_water',
    'cld_opd_mlclouds_ice',
    'cloud_type',
    'flag',
    'cld_opd_dcomp',
    'cld_opd_mlclouds',
]


feature_sets = [features, features[:-6], features[:-1], features[:-2]]
n_estimators = [10, 25, 50, 75, 100, 500, 1000, 1500, 2000, 2500, 3000]
max_depth = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

param_dict = {}
i=0
for n in n_estimators:
    for m in max_depth:
        for f in feature_sets:
            param_dict[i] = {'n_estimators': n,
                             'max_depth': m,
                             'features': f}
            i+=1
            
def plot_cm(cm, title='Confusion Matrix'):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.yaxis.set_ticklabels(['clear', 'water', 'ice'])
    ax.xaxis.set_ticklabels(['clear', 'water', 'ice'])
    plt.show()

    
def plot_binary_cm(cm, title='Confusion Matrix'):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.yaxis.set_ticklabels(['clear', 'cloudy'])
    ax.xaxis.set_ticklabels(['clear', 'cloudy'])
    plt.show()   
    

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    array_string = array_string.replace('\n', ',')
    return np.array(ast.literal_eval(array_string))


def load_data(file_name=f'{working_dir}/mlclouds_all_data.csv', encode_flag=True):
    
    df = pd.read_csv(file_name)
    
    if encode_flag:
        #label_encoder = LabelEncoder()
        df['flag'] = df['flag'].replace({'clear': 0, 'bad_cloud': 3, 'water_cloud': 1, 'ice_cloud': 2})
        #label_encoder.fit_transform(df['flag'])
    return df


def int_or_none(arg):
    if str(arg) == 'None':
        return None
    else:
        try:
            arg = int(arg)
            return arg
        except:
            raise TypeError
    

def sample_df(df, samples=None):
    if samples is not None:
        df_samp = df.sample(n=samples)
    else:
        df_samp = df
    return df_samp    

def output_new_csv(test_size=0.2, n_estimators=75, max_depth=5,
                   features=features[:-6], samples=None):
    
    print(f'started loading data {time.ctime()}')
    
    df = load_data()
    df = sample_df(df, samples)
    
    print(f'finished loading data {time.ctime()}')
    X_train, X_test, y_train, y_test, indices_train, indices_test = split_data(df, features, test_size) 
    
    title = get_model_title(max_depth=max_depth, n_estimators=n_estimators,
                            features=features)
    
    pipe = train_model(
        X_train, y_train,
        samples=samples, max_depth=max_depth,
        n_estimators=n_estimators,
        features=features, test_size=test_size,
        save_model=True, model_name=title)
    
    y_pred_all = np.zeros((len(indices_train) + len(indices_test)))
    y_pred_all[indices_train] = pipe.predict(X_train)
    y_pred_all[indices_test] = pipe.predict(X_test)
    mask = np.zeros((len(y_pred_all)))
    mask[indices_train] = 0
    mask[indices_test] = 1
    
    invert_cloud_map = {v: k for k, v in cloud_map.items()}
    y_pred_all = np.array([invert_cloud_map[v] for v in y_pred_all])
    
    print(f'updating csv with xgb fields {time.ctime()}')
    
    df['cld_opd_xgb'] = 0
    df['cld_reff_xgb'] = 0
    df['cloud_type_xgb'] = 0
    ice_mask = y_pred_all == 'ice'
    water_mask = y_pred_all == 'water'
    df.loc[ice_mask, 'cld_opd_xgb'] = df.loc[ice_mask, 'cld_opd_mlclouds_ice']
    df.loc[ice_mask, 'cld_reff_xgb'] = df.loc[ice_mask, 'cld_reff_mlclouds_ice']
    df.loc[water_mask, 'cld_opd_xgb'] = df.loc[water_mask, 'cld_opd_mlclouds_water']
    df.loc[water_mask, 'cld_reff_xgb'] = df.loc[water_mask, 'cld_reff_mlclouds_water']
    df.loc[ice_mask, 'cloud_type_xgb'] = 6
    df.loc[water_mask, 'cloud_type_xgb'] = 2

    print(f'done updating {time.ctime()}')        

    df['cloud_id_xgb'] = y_pred_all
    df['mask_xgb'] = mask

    df['flag'] = df['flag'].replace({0: 'clear', 1: 'water_cloud', 2: 'ice_cloud', 3: 'bad_cloud'})

    csv_file = os.path.join(output_dir, f'{title}_data.csv')
    print(f'writing csv: {csv_file}')
    
    df.to_csv(csv_file)

def select_features(df, features):    
    X = df[features]
    y = df['nom_cloud_id']
    y = y.replace(cloud_map)
    return X, y


def split_data(df, features, test_size):
    X, y = select_features(df, features)
    return train_test_split(X, y, np.arange(X.shape[0]), test_size=test_size)   


def plot_wisconsin_cm():

    df = load_data(encode_flag=False)
    y_wisc = df['flag']
    index = y_wisc.index[y_wisc != 'bad_cloud'].tolist()
    y_wisc = y_wisc.iloc[index]
    y_wisc = y_wisc.replace({'clear': 0, 'water_cloud': 1, 'ice_cloud': 2})
    y = df['nom_cloud_id']
    y = y.replace(cloud_map)
    y_trim = y.iloc[index]
    
    cm = confusion_matrix(y_trim, y_wisc)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plot_cm(cm, 'Wisconsin Confusion Matrix')
    
def plot_wisconsin_binary_cm():

    df = load_data(encode_flag=False)
    y_wisc = df['flag']
    index = y_wisc.index[y_wisc != 'bad_cloud'].tolist()
    y_wisc = y_wisc.iloc[index]
    y_wisc = y_wisc.replace({'clear': 0, 'water_cloud': 1, 'ice_cloud': 1})
    y = df['nom_cloud_id']
    y = y.replace({'clearsky': 0, 'water': 1, 'ice': 1})
    y_trim = y.iloc[index]
    
    cm = confusion_matrix(y_trim, y_wisc)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plot_binary_cm(cm, 'Wisconsin Confusion Matrix')

    
def train_model(X_train, y_train, samples=None, max_depth=10,
                n_estimators=500, features=features[:-1],
                test_size=0.2, save_model=True, model_name='model'):
    
    start_time = time.time()
    print(f'started training {time.ctime()}')
    
    print(f'using max_depth={max_depth}, '
          f'n_estimators={n_estimators}, '
          f'n_features={len(features)}, '
          f'test_size={test_size}, ' 
          f'samples={samples}')
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric='merror')
    
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', model)])
    
    pipe.fit(X_train, y_train)
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'finished training {time.ctime()}')
    print(f'time elapsed: {time_elapsed}')
    
    model_file = os.path.join(output_dir, f'{model_name}.pkl')
    if save_model:
        joblib.dump(pipe, model_file)
        print(f'saved model: {model_file}')
    
    return pipe

def get_model_title(max_depth=5, n_estimators=75, features=features[:-2]):
    title = f'xgb_max_depth_{max_depth}'
    title += f'_n_estimators_{n_estimators}_n_features_{len(features)}'
    return title

def get_architecture_info(pipe, X_train, X_test, y_train, y_test, features):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    mae = mae_perc(y_test, y_pred)
            
    y_pred_binary = np.array([0 if y==0 else 1 for y in y_pred])
    y_test_binary = np.array([0 if y==0 else 1 for y in y_test])
            
    binary_cm = confusion_matrix(y_test_binary, y_pred_binary)
    binary_cm = binary_cm.astype('float') / binary_cm.sum(axis=1)[:, np.newaxis]
            
    params = pipe['model'].get_params()
    n_estimators = params.get('n_estimators', None)
    max_depth = params.get('max_depth', None)
            
    title = get_model_title(max_depth=max_depth, n_estimators=n_estimators,
                            features=features)
    mean_accuracy = np.mean([cm[i][i] for i in range(cm.shape[0])])
    binary_mean_accuracy = np.mean([binary_cm[i][i] for i in range(binary_cm.shape[0])])
    val_loss = log_loss(y_test, pipe.predict_proba(X_test))
    train_loss = log_loss(y_train, pipe.predict_proba(X_train))
    diff_loss = np.abs(val_loss - train_loss)
    ratio_loss = diff_loss / np.mean([val_loss, train_loss])
    mean_loss = np.mean([ratio_loss, val_loss])
    score = pipe.score(X_test, y_test)
    new_row = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'title': title,
        'model': pipe['model'].__class__.__name__,
        'features': len(features),                       
        'mae': mae,
        'score': score,
        'confusion_matrix': np.array(cm),
        'binary_confusion_matrix': np.array(binary_cm),
        'mean_accuracy': mean_accuracy,
        'binary_mean_accuracy': binary_mean_accuracy,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'diff_loss': diff_loss,
        'ratio_loss': ratio_loss,
        'mean_loss': mean_loss}
    return new_row


def output_model_info_csv(samples=None, test_size=0.2):
    
    df = load_data()
    
    if samples is not None:
        df_samp = df.sample(n=samples)
    else:
        df_samp = df

    model_info = pd.DataFrame()
    
    models = []

    global n_estimators
    global max_depth
    global feature_sets
    
    for n in n_estimators:
        for m in max_depth:
            models.append(xgb.XGBClassifier(n_estimators=n, max_depth=m, use_label_encoder=False, eval_metric='merror'))
    
    for f in feature_sets:
        
        X_train, X_test, y_train, y_test, _, _ = split_data(df_samp, f, test_size)
        
        for model in models:

            pipe = Pipeline([('scaler', StandardScaler()),
                             ('model', model)])
  

            pipe.fit(X_train, y_train)
            
            new_row = get_architecture_info(pipe, X_train, X_test, y_train, y_test, f)
            model_info = model_info.append(new_row, ignore_index=True)
            print(f'Added {new_row["title"]}')
    
    csv_file = os.path.join(output_dir, 'model_info.csv')
    print(f'writing csv: {csv_file}')
    model_info.to_csv(csv_file)
    
def batch_run(samples=None, n_estimators=500, max_depth=20, test_size=0.2, features=features[:-1]):
    
    df = load_data()
    
    if samples is not None:
        df_samp = df.sample(n=samples)
    else:
        df_samp = df

    model_info = pd.DataFrame()
    
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, use_label_encoder=False, eval_metric='merror')
        
    X_train, X_test, y_train, y_test, _, _ = split_data(df_samp, features, test_size)
        
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', model)])
  
    pipe.fit(X_train, y_train)
    new_row = get_architecture_info(pipe, X_train, X_test, y_train, y_test, features)
    model_info = model_info.append(new_row, ignore_index=True)
    print(f'Added {new_row["title"]}')
    print(f'mean_accuracy: {new_row["mean_accuracy"]}')
    print(f'binary_mean_accuracy: {new_row["binary_mean_accuracy"]}')
    print(f'score: {new_row["score"]}')
    print(f'val_loss: {new_row["val_loss"]}')  
    print(f'train_loss: {new_row["train_loss"]}')  
    print(f'diff_loss: {new_row["diff_loss"]}')   
    
    if os.path.exists(batch_model_info_file):
        model_info.to_csv(batch_model_info_file, mode='a', header=False)
    else:
        model_info.to_csv(batch_model_info_file)
    
def load_model_info_csv(file_name=f'{output_dir}/batch_model_info.csv'):
    df = pd.read_csv(file_name)
    df['confusion_matrix'] = df['confusion_matrix'].apply(from_np_array)
    df['binary_confusion_matrix'] = df['binary_confusion_matrix'].apply(from_np_array)
    return df
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Cloud Classification methods')
    parser.add_argument('--model_info', action='store_true')
    parser.add_argument('--new_csv', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--batch_run', action='store_true')
    parser.add_argument('--batch_search', action='store_true')
    parser.add_argument('-samples', default=None, type=int_or_none)
    parser.add_argument('-param_id', default=None, type=int_or_none)
    args = parser.parse_args()
    
    if args.model_info:
        output_model_info_csv(samples=args.samples)
    
    if args.new_csv:
        output_new_csv(samples=args.samples)

    if args.grid_search:
        tune_parameters(samples=args.samples)
        
    if args.batch_run:
        params = param_dict[args.param_id]
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        features = params['features']
        batch_run(n_estimators=n_estimators,
                  max_depth=max_depth,
                  features=features,
                  samples=args.samples)
        
    if args.batch_search:
        os.system(f'rm -f {batch_model_info_file}')
        print(f'removing previous info file: {batch_model_info_file}')
        for id_ in param_dict:
            os.system(f'sbatch ./batch_script.sh {id_} {args.samples}')
            
