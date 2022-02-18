import numpy as np
import pandas as pd
from nsrdb.all_sky.all_sky import all_sky, ALL_SKY_ARGS

if __name__ == '__main__':
    df = pd.read_csv('./mlclouds_all_data_xgb.csv', index_col=0)

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
        dset_out = dset if cloud_phase is None else f'{cloud_phase}_{dset}'
        df[dset_out] = out[dset].flatten()

    df.to_csv('./mlclouds_all_data_xgb_results.csv')
