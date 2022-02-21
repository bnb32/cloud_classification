import numpy as np
import pandas as pd
import os
from nsrdb.all_sky.all_sky import all_sky, ALL_SKY_ARGS

import cloud_classification as cc

if __name__ == '__main__':
    xgb_csv = os.path.join(cc.output_dir, 'mlclouds_all_data_xgb.csv')
    df = pd.read_csv(xgb_csv, index_col=0)

    ignore = ('cloud_fill_flag',)

    all_sky_args = [dset for dset in ALL_SKY_ARGS if dset not in ignore]
    all_sky_input = {dset: df[dset].values for dset in all_sky_args}

    all_sky_input['cloud_type'] = df['cloud_type_xgb'].values
    all_sky_input['cld_opd_dcomp'] = df['cld_opd_xgb'].values
    all_sky_input['cld_reff_dcomp'] = df['cld_reff_xgb'].values
    all_sky_input = {k: np.expand_dims(v, axis=1)
                     for k, v in all_sky_input.items()}
    all_sky_input['time_index'] = df['time_index'].values

    out = all_sky(**all_sky_input)

    for dset in ('ghi', 'dni', 'dhi'):
        df[f'xgb_results_{dset}'] = out[dset].flatten()

    output_xgb_csv = os.path.join(cc.output_dir, 'mlclouds_all_data_xgb_results.csv')    
    df.to_csv(output_xgb_csv)
