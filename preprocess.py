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

@hydra.main(config_path="configs/preprocess")
def main(cfg: DictConfig) -> None:
    cont_features = OmegaConf.to_object(cfg.cont_features)
    cat_features = OmegaConf.to_object(cfg.cat_features)
    misc_features = OmegaConf.to_object(cfg.misc_features)
    input_branches = cont_features + cat_features + misc_features
    input_path = fill_placeholders(cfg.input_path, {'{year}': cfg.year})
    output_path = fill_placeholders(cfg.output_path, {'{year}': cfg.year})

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
        input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})
        print(f'opening {sample_name}...')
        with uproot.open(f'{input_path}/{input_filename}') as f:
            if cfg.for_training:
                processes = list(sample.values())[0]
                for process_name, process_cfg in processes.items():
                    print(f'    loading {process_name}...')
                    data_sample = f[cfg.input_tree_name].arrays(input_branches, cut=process_cfg['cut'], library='pd')
                    data_sample['group_name'] = process_name
                    data_sample[_target] = process_cfg['class']
                    data_samples.append(data_sample)
            else:
                data_sample = f[cfg.input_tree_name].arrays(input_branches, cut=None, library='pd')
                data_sample['group_name'] = sample_name
                data_sample[_target] = -1
                data_samples.append(data_sample)
    data = pd.concat(data_samples, ignore_index=True)
    del(data_samples); gc.collect()

    # some preprocessing
    data.replace([np.inf, -np.inf], np.nan, inplace=True) # lumin handles nans automatically
    data['njets'] = data.njets.clip(0, 5)

    # split data into output nodes: either train+test (for training) or sample_name based splitting (for prediction)
    if cfg.for_training:
        # derive key for stratified split
        strat_key = 'strat_key'
        data[strat_key] = ids2unique(data[[_target] + cat_features].values);
        # output_samples[0] -> train, output_samples[1] -> test
        output_samples = train_test_split(data, train_size=cfg.train_size, stratify=data[strat_key], random_state=1357)
        output_sample_names = cfg.output_samples
        assert len(output_sample_names)==len(output_samples)
        input_pipe = fit_input_pipe(output_samples[0], cont_features, to_absolute_path(f'{output_path}/{cfg.pipe_name}'), norm_in=cfg.norm, pca=cfg.pca)
        cat_maps, cat_szs = proc_cats(output_samples[0], cat_features, output_samples[1])
    else:
        strat_key = None # no stratification for prediction
        output_sample_names, output_samples = data.groupby('group_name')
        output_sample_names = [fill_placeholders(cfg.output_filename_template, {'{sample_name}': n, '{year}': cfg.year}) for n in output_sample_names]
        with open(cfg.input_pipe_file, 'rb') as f:
            input_pipe = pickle.load(f)

    # loop over output nodes and store each into a fold file
    for output_sample, output_sample_name in zip(output_samples, output_sample_names):
        # weights section
        if cfg.for_training:
            # add training weights accounting for imbalance in data
            output_sample['w_class_imbalance'] = 1
            for class_label in set(data[_target]):
                output_sample.loc[output_sample[_target] == class_label, 'w_class_imbalance'] = data.shape[0]/data.query(f'{_target}=={class_label}').shape[0]

            # add training weights as used in CP in HTT analysis
            output_sample['class_weight'] = 1 # these are derived per each class as: sum("weight") for whole dataset / sum("weight") for class
            for class_label in set(data[_target]):
                output_sample.loc[output_sample[_target] == class_label, 'class_weight'] = np.sum(data['weight'])/np.sum(data.loc[data[_target] == class_label, 'weight'])
            output_sample['w_cp'] = abs(output_sample['weight'])*output_sample['class_weight']
            weight_features = ['w_class_imbalance', 'w_cp', 'class_weight', 'weight']
        else:
            weight_features = None

        # apply normalisation
        output_sample[cont_features] = input_pipe.transform(output_sample[cont_features])

        # store into a hdf5 fold file
        df2foldfile(df=output_sample,
                    n_folds=cfg.n_folds, strat_key=strat_key,
                    cont_feats=cont_features,
                    cat_feats=cat_features, cat_maps=cat_maps,
                    targ_feats=_target, targ_type='int',
                    wgt_feat=None,
                    misc_feats=weight_features,
                    savename=to_absolute_path(f'{output_path}/{output_sample_name}')
                    )

if __name__ == '__main__':
    main()
