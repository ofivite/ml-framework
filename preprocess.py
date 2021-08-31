import uproot

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from lumin.utils.misc import ids2unique
from lumin.data_processing.pre_proc import fit_input_pipe, proc_cats
from lumin.data_processing.file_proc import df2foldfile

import os
import pickle
import gc
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from utils.processing import fill_placeholders

@hydra.main(config_path="configs/preprocess")
def main(cfg: DictConfig) -> None:
    cont_features = OmegaConf.to_object(cfg.cont_features)
    cat_features = OmegaConf.to_object(cfg.cat_features)
    misc_features = OmegaConf.to_object(cfg.misc_features)
    input_branches = OmegaConf.to_object(cfg.input_branches)
    output_path = to_absolute_path(cfg.output_path)
    os.makedirs(output_path, exist_ok=True)

    # combine all input data nodes into a single pandas dataframe
    data_samples = []
    _target = 'target' # internal target name
    for sample in cfg.input_samples:
        if cfg.for_training:
            assert len(sample)==1 # for training, sample is a dictionary with 1 element, see the cfg file
            sample_name = list(sample.keys())[0]
        else:
            assert type(sample)==str
            sample_name = sample
        input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name})
        print(f'\n--> Opening {sample_name}...')
        with uproot.open(f'{cfg.input_path}/{input_filename}') as f:
            if cfg.for_training:
                processes = list(sample.values())[0]
                for process_name, process_cfg in processes.items():
                    print(f'    loading {process_name}')
                    data_sample = f[cfg.input_tree_name].arrays(input_branches, cut=process_cfg['cut'], library='pd')
                    data_sample['group_name'] = process_name
                    data_sample[_target] = process_cfg['class']
                    data_samples.append(data_sample)
            else:
                data_sample = f[cfg.input_tree_name].arrays(input_branches, cut=None, library='pd')
                data_sample['group_name'] = sample_name
                data_sample[_target] = -1
                data_samples.append(data_sample)
    print('\n--> Combining inputs together')
    data = pd.concat(data_samples, ignore_index=True)
    del(data_samples); gc.collect()

    print('\n--> Preprocessing')
    # check for NaNs and infs presence
    data_nans = data.isna()
    data_infs = data.isin([-np.inf, np.inf])
    if (nan_sum := np.sum(data_nans.values)) > 0:
        raise Exception(f'\nFound {nan_sum} NaNs in columns: {data_nans.columns[data_nans.any(axis=0)].tolist()}. Please take care of preprocessing them.')
    if (inf_sum := np.sum(data_infs.values)) > 0:
        raise Exception(f'\nFound {inf_sum} inf values in columns: {data_infs.columns[data_infs.any(axis=0)].tolist()}. Please take care of preprocessing them.')

    # clip tails in njets
    data['njets'] = data.njets.clip(0, 5)

    # process categorical features to be valued 0->cardinality-1
    # TODO: think if doing this transformation jointly may cause troubles
    cat_maps, cat_szs = proc_cats(data, cat_features)

    # split data into output nodes: either train+test (for training) or sample_name based splitting (for prediction)
    if cfg.for_training:
        # derive key for stratified split
        strat_key = 'strat_key'
        data[strat_key] = ids2unique(data[[_target] + cat_features].values)

        # split into output_samples[0] -> train, output_samples[1] -> test
        output_samples = train_test_split(data, train_size=cfg.train_size, stratify=data[strat_key], random_state=1357)
        output_sample_names = cfg.output_samples
        assert len(output_sample_names)==len(output_samples)

        # fit scaling pipe
        input_pipe = fit_input_pipe(output_samples[0], cont_features, f'{output_path}/{cfg.pipe_name}', norm_in=cfg.norm, pca=cfg.pca)
    else:
        strat_key = None # no stratification for prediction
        outputs = {name: group for name, group in data.groupby('group_name')}
        output_samples = outputs.values()
        output_sample_names = outputs.keys()
        output_sample_names = [fill_placeholders(cfg.output_filename_template, {'{sample_name}': n}) for n in output_sample_names]

        # fetch already fitted pipe
        with open(to_absolute_path(cfg.input_pipe_file), 'rb') as f:
            input_pipe = pickle.load(f)

    # derive training weights as of CP analysis (based on the whole input data)
    if cfg.for_training and cfg.compute_w_CP:
        class_weight_map = {}
        for class_label in set(data[_target]):
            class_weight_map[class_label] = np.sum(data['weight'])/np.sum(data.query(f'{_target} == {class_label}')['weight'])

    # loop over output nodes and store each into a fold file
    print('\n--> Storing to output files...')
    for output_sample_name, output_sample in zip(output_sample_names, output_samples):
        print(f'    {output_sample_name}')
        # derive class imbalance weights (per output node: train/test)
        if cfg.for_training:
            w_class_imbalance_map = {}
            for class_label in set(output_sample[_target]):
                w_class_imbalance_map[class_label] = len(output_sample)/len(output_sample.query(f'{_target}=={class_label}'))

            # add training weights accounting for imbalance in data
            output_sample['w_class_imbalance'] = output_sample[_target].map(w_class_imbalance_map)
            if cfg.compute_w_CP:
                output_sample['class_weight'] = output_sample[_target].map(class_weight_map)
                output_sample['w_cp'] = abs(output_sample['weight'])*output_sample['class_weight']

        # apply normalisation
        output_sample[cont_features] = input_pipe.transform(output_sample[cont_features])

        # store into a hdf5 fold file
        df2foldfile(df=output_sample,
                    n_folds=cfg.n_lumin_folds, strat_key=strat_key,
                    cont_feats=cont_features,
                    cat_feats=cat_features, cat_maps=cat_maps,
                    targ_feats=_target, targ_type='int',
                    wgt_feat=None,
                    misc_feats=misc_features,
                    savename=f'{output_path}/{output_sample_name}'
                    )

if __name__ == '__main__':
    main()
