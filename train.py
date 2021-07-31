import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

from lumin.nn.data.fold_yielder import FoldYielder

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from utils.processing import fill_placeholders
from utils.plotting import plot_class_score

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train_file = fill_placeholders(to_absolute_path(cfg.train_file), {'{year}': cfg.year})
    test_file = fill_placeholders(to_absolute_path(cfg.test_file), {'{year}': cfg.year})
    input_pipe_file = fill_placeholders(to_absolute_path(cfg.input_pipe_file), {'{year}': cfg.year})
    print(f'\n[INFO] Will split training data set into folds over values of ({cfg.xtrain_split_feature}) feature with number of splits ({cfg.n_splits})')

    # fetch feature/weight/target names
    train_features = cfg.cont_features + cfg.cat_features # features to be used in training
    weight_name = cfg.weight_name
    target_name = 'gen_target' # internal target name defined inside of lumin

    # prepare train/test data
    print('\n--> Loading training data')
    train_fy = FoldYielder(train_file, input_pipe=input_pipe_file)
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False, verbose=False, suppress_warn=True)
    train_df['w_cp'] = train_fy.get_column('w_cp')
    train_df['w_class_imbalance'] = train_fy.get_column('w_class_imbalance')
    train_df['plot_weight'] = train_fy.get_column('weight')
    train_df[cfg.xtrain_split_feature] = train_fy.get_column(cfg.xtrain_split_feature)
    train_df['fold_id'] = train_df[cfg.xtrain_split_feature] % cfg.n_splits
    #
    print('\n--> Loading testing data')
    test_fy = FoldYielder(test_file, input_pipe=input_pipe_file)
    test_df = test_fy.get_df(inc_inputs=True, deprocess=False, verbose=False, suppress_warn=True)
    test_df['plot_weight'] = test_fy.get_column('weight')

    # check that there is no more that 5% difference between folds in terms of number of entries
    fold_id_count_diff = np.std(train_df['fold_id'].value_counts()) / np.mean(train_df['fold_id'].value_counts())
    if fold_id_count_diff > 0.05:
        raise Exception(f'Observed {fold_id_count_diff * 100}% relative difference in number of entries across folds. Please check that the split is done equally.')

    logo = LeaveOneGroupOut() # splitter into folds for training/validation
    with mlflow.start_run():
        # enable auto logging for mlflow & log some cfg parameters
        mlflow.lightgbm.autolog(log_models=False) # models are logged separately for each fold
        mlflow.log_params({
            'train_file': train_file,
            'test_file': test_file,
            'input_pipe_file': input_pipe_file,
            'xtrain_split_feature': cfg.xtrain_split_feature,
            'weight_name': cfg.weight_name
        })

        print(f'\n--> Training model...')
        for i_fold, (train_idx, validation_idx) in enumerate(logo.split(train_df, groups=train_df['fold_id'])):
            print(f'\n    on all folds except fold {i_fold}\n\n')
            train_fold_df = train_df.iloc[train_idx]
            validation_fold_df = train_df.iloc[validation_idx]

            # check that `i_fold` is the same as fold ID corresponding to each fold split
            validation_fold_idx = set(validation_fold_df['fold_id'])
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
