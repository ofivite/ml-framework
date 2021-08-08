import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

from lumin.nn.data.fold_yielder import FoldYielder

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from utils.processing import fill_placeholders

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # load training data into DataFrame + add necessary columns
    print('\n--> Loading training data')
    train_file = fill_placeholders(to_absolute_path(cfg.train_file), {'{year}': cfg.year})
    train_fy = FoldYielder(train_file)
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False, nan_to_num=False, verbose=False, suppress_warn=True)
    train_df['w_cp'] = train_fy.get_column('w_cp')
    train_df['w_class_imbalance'] = train_fy.get_column('w_class_imbalance')

    # define feature/weight/target names
    train_features = cfg.cont_features + cfg.cat_features # features to be used in training
    weight_name = cfg.weight_name
    target_name = 'gen_target' # internal target name defined inside of lumin library
    fold_id_column = 'fold_id'

    if cfg.n_splits > 1:
        assert type(cfg.n_splits)==int
        split_feature_values = train_fy.get_column(cfg.xtrain_split_feature)
        train_df[fold_id_column] = (split_feature_values % cfg.n_splits).astype('int32')

        # check that there is no more that 5% difference between folds in terms of number of entries
        fold_id_count_diff = np.std(train_df[fold_id_column].value_counts()) / np.mean(train_df[fold_id_column].value_counts())
        if fold_id_count_diff > 0.05:
            raise Exception(f'Observed {fold_id_count_diff * 100}% relative difference in number of entries across folds. Please check that the split is done equally.')

        print(f'\n[INFO] Will split training data set into ({cfg.n_splits}) folds over values of ({cfg.xtrain_split_feature}) feature to perform cross-training')
        splitter = LeaveOneGroupOut()
        idx_yielder = splitter.split(train_df, groups=train_df[fold_id_column])
    elif cfg.n_splits == 1:
        print(f'\n[INFO] Will train a single model on ({cfg.train_size}) part of the training data set with the rest used for validation')
        train_df[fold_id_column] = 0
        splitter = ShuffleSplit(n_splits=1, train_size=cfg.train_size, random_state=1357)
        idx_yielder = splitter.split(train_df)
    else:
        raise ValueError(f'n_splits should be positive integer, got {cfg.n_splits}')

    with mlflow.start_run():
        # enable auto logging for mlflow & log some cfg parameters
        mlflow.lightgbm.autolog(log_models=False)
        mlflow.log_params({
            'train_file': train_file,
            'xtrain_split_feature': cfg.xtrain_split_feature,
            'weight_name': cfg.weight_name,
            'target_name': target_name
        })

        print(f'\n--> Training model...')
        for i_fold, (train_idx, validation_idx) in enumerate(idx_yielder):
            if cfg.n_splits > 1: print(f'\n\n    leaving fold {i_fold} out\n\n')
            train_fold_df = train_df.iloc[train_idx]
            validation_fold_df = train_df.iloc[validation_idx]

            # check that `i_fold` is the same as fold ID corresponding to a validation fold
            validation_fold_idx = set(validation_fold_df[fold_id_column])
            assert len(validation_fold_idx)==1 and i_fold in validation_fold_idx

            # construct lightgbm dataset
            train_data = lgb.Dataset(train_fold_df[train_features],
                                     label=train_fold_df[target_name],
                                     weight=train_fold_df[weight_name])
            validation_data = lgb.Dataset(validation_fold_df[train_features],
                                          label=validation_fold_df[target_name],
                                          weight=validation_fold_df[weight_name],
                                          reference=train_data)

            # train booster
            model = lgb.train(OmegaConf.to_object(cfg.model_param),
                              train_data,
                              valid_sets=[train_data, validation_data], valid_names=[f'train_{i_fold}', f'valid_{i_fold}'])

            # infer signature of the model and log into mlflow
            signature = infer_signature(train_fold_df[train_features], model.predict(train_fold_df[train_features]))
            mlflow.lightgbm.log_model(model, f'model_{i_fold}', signature=signature, input_example=train_fold_df.iloc[0][train_features].to_numpy())
            # mlflow.log_artifact(train_idx)

if __name__ == '__main__':
    main()
