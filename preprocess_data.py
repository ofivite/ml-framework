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
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string

@hydra.main(config_path="configs", config_name="preprocess_cfg")
def main(cfg: DictConfig) -> None:

    cat_features = OmegaConf.to_object(cfg.cat_features)
    cont_features = OmegaConf.to_object(cfg.cont_features)

    # combine all data samples into single pandas dataframe
    data_samples = {}
    for sample_name, sample_path in cfg.sample_paths.items():
        sample_path = fill_placeholders(sample_path, {'{year}': cfg.year})
        print(f'loading {sample_name}...')
        data_samples[sample_name] = uproot.open(sample_path)[cfg.tree_name].arrays(cfg.branches, library='pd')
        data_samples[sample_name]['sample_name'] = sample_name
    data = pd.concat(data_samples, ignore_index=True)

    # assign target label to each sample according to a map from cfg file
    target = cfg.target_name
    data[target] = data['sample_name'].map(cfg.sample_to_class)
    data.drop(columns='sample_name', inplace=True)

    # some preprocessing
    data.replace([np.inf, -np.inf], np.nan, inplace=True) # lumin handles nans automatically
    data['njets'] = data.njets.clip(0, 5)

    # derive key for stratified split
    data['strat_key'] = ids2unique(data[[target] + cat_features].values)

    # split into train and test samples
    train_df, test_df = train_test_split(data, train_size=cfg.train_size, stratify=data['strat_key'], random_state=1357)

    # apply normalisation
    input_pipe = fit_input_pipe(train_df, cont_features, to_absolute_path(f'{cfg.output_folder}/{cfg.pipe_name}'), norm_in=cfg.norm, pca=cfg.pca)
    train_df[cont_features] = input_pipe.transform(train_df[cont_features])
    test_df[cont_features] = input_pipe.transform(test_df[cont_features])
    cat_maps, cat_szs = proc_cats(train_df, cat_features, test_df)

    # derive weights accounting for imbalance in data
    w_scaling_mult = 1e6
    weight = cfg.weight_name
    train_df[weight], test_df[weight] = 1, 1
    for class_label in set(cfg.sample_to_class.values()):
        train_df.loc[train_df[target] == class_label, weight] *= (w_scaling_mult/np.sum(train_df.loc[train_df[target] == class_label, weight]))
        test_df.loc[test_df[target] == class_label, weight] *= (w_scaling_mult/np.sum(test_df.loc[test_df[target] == class_label, weight]))

    # check_val_set(train_df[train_features], val_df[train_features], test_df[train_features])

    # store into a hdf5 fold file
    df2foldfile(df=train_df,
                n_folds=cfg.n_folds, strat_key='strat_key',
                cont_feats=cont_features,
                cat_feats=cat_features, cat_maps=cat_maps,
                targ_feats=target, targ_type='int',
                wgt_feat=weight,
                savename=to_absolute_path(f'{cfg.output_folder}/{cfg.train_name}')
                )

    df2foldfile(df=test_df,
                n_folds=cfg.n_folds,
                cont_feats=cont_features,
                cat_feats=cat_features, cat_maps=cat_maps,
                targ_feats=target, targ_type='int',
                wgt_feat=weight,
                savename=to_absolute_path(f'{cfg.output_folder}/{cfg.test_name}')
                )

if __name__ == '__main__':
    main()
