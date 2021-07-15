import uproot

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from lumin.utils.misc import ids2unique
from lumin.utils.data import check_val_set
from lumin.data_processing.pre_proc import fit_input_pipe, proc_cats
from lumin.data_processing.file_proc import df2foldfile
from lumin.nn.data.fold_yielder import FoldYielder

import pickle
import gc
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from utils.processing import fill_placeholders

@hydra.main(config_path="configs", config_name="preprocess_cfg")
def main(cfg: DictConfig) -> None:
    cont_features = OmegaConf.to_object(cfg.cont_features)
    cat_features = OmegaConf.to_object(cfg.cat_features)
    misc_features = OmegaConf.to_object(cfg.misc_features)
    input_branches = cont_features + cat_features + misc_features
    input_path = fill_placeholders(cfg.input_path, {'{year}': cfg.year})
    output_path = fill_placeholders(cfg.output_path, {'{year}': cfg.year})

    # combine all data samples into single pandas dataframe
    data_samples = {}
    _target = 'target' # internal target name
    for sample_name, processes in cfg.input_samples.items():
        print(f'opening {sample_name}...')
        input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})
        with uproot.open(f'{input_path}/{input_filename}') as f:
            for process_name, process_cfg in processes.items():
                print(f'    loading {process_name}...')
                data_samples[process_name] = f[cfg.input_tree_name].arrays(input_branches, cut=process_cfg['cut'], library='pd')
                data_samples[process_name][_target] = process_cfg['class']
                data_samples[process_name]['process'] = process_name
    data = pd.concat(data_samples, ignore_index=True)
    del(data_samples); gc.collect()

    # some preprocessing
    data.replace([np.inf, -np.inf], np.nan, inplace=True) # lumin handles nans automatically
    data['njets'] = data.njets.clip(0, 5)

    # derive key for stratified split
    data['strat_key'] = ids2unique(data[[_target] + cat_features].values)

    # split into train and test samples
    train_df, test_df = train_test_split(data, train_size=cfg.train_size, stratify=data['strat_key'], random_state=1357)

    # apply normalisation
    input_pipe = fit_input_pipe(train_df, cont_features, to_absolute_path(f'{output_path}/{cfg.pipe_name}'), norm_in=cfg.norm, pca=cfg.pca)
    train_df[cont_features] = input_pipe.transform(train_df[cont_features])
    test_df[cont_features] = input_pipe.transform(test_df[cont_features])
    cat_maps, cat_szs = proc_cats(train_df, cat_features, test_df)

    # training weights accounting for imbalance in data
    train_df['w_class_imbalance'], test_df['w_class_imbalance'] = 1, 1
    for class_label in set(data[_target]):
        train_df.loc[train_df[_target] == class_label, 'w_class_imbalance'] = data.shape[0]/data.query(f'{_target}=={class_label}').shape[0]
        test_df.loc[test_df[_target] == class_label, 'w_class_imbalance'] = data.shape[0]/data.query(f'{_target}=={class_label}').shape[0]

    # training weights as used in CP in HTT analysis
    train_df['class_weight'], test_df['class_weight'] = 1, 1 # these are derived per each class as: sum("weight") for whole dataset / sum("weight") for class
    for class_label in set(data[_target]):
        train_df.loc[train_df[_target] == class_label, 'class_weight'] = np.sum(data['weight'])/np.sum(data.loc[data[_target] == class_label, 'weight'])
        test_df.loc[test_df[_target] == class_label, 'class_weight'] = np.sum(data['weight'])/np.sum(data.loc[data[_target] == class_label, 'weight'])
    train_df['w_cp'] = abs(train_df['weight'])*train_df['class_weight']
    test_df['w_cp'] = abs(test_df['weight'])*test_df['class_weight']

    # store into a hdf5 fold file
    df2foldfile(df=train_df,
                n_folds=cfg.n_folds, strat_key='strat_key',
                cont_feats=cont_features,
                cat_feats=cat_features, cat_maps=cat_maps,
                targ_feats=_target, targ_type='int',
                wgt_feat=None,
                misc_feats=['w_class_imbalance', 'w_cp', 'class_weight', 'weight'],
                savename=to_absolute_path(f'{output_path}/{cfg.train_name}')
                )

    df2foldfile(df=test_df,
                n_folds=cfg.n_folds, strat_key='strat_key',
                cont_feats=cont_features,
                cat_feats=cat_features, cat_maps=cat_maps,
                targ_feats=_target, targ_type='int',
                wgt_feat=None,
                misc_feats=['w_class_imbalance', 'w_cp', 'class_weight', 'weight'],
                savename=to_absolute_path(f'{output_path}/{cfg.test_name}')
                )

if __name__ == '__main__':
    main()
