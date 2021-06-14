import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

import uproot
from lumin.utils.misc import ids2unique
from lumin.utils.data import check_val_set
from lumin.data_processing.pre_proc import fit_input_pipe, proc_cats
from lumin.data_processing.file_proc import df2foldfile
from lumin.nn.data.fold_yielder import FoldYielder

# import click
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@hydra.main(config_path=".", config_name="training_cfg")
def main(cfg: DictConfig) -> None:
    # enable auto logging
    mlflow.lightgbm.autolog()

    # feature names to be used in training
    train_features = cfg.cont_features + cfg.cat_features
    weight_feature = 'gen_weight'
    target_feature = 'gen_target'

    # prepare train/test data
    train_fy = FoldYielder(to_absolute_path(cfg.train_file), input_pipe=to_absolute_path(cfg.pipe_file))
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False)
    train_data = lgb.Dataset(train_df[train_features], label=train_df[target_feature], weight=train_df[weight_feature])
    #
    test_fy = FoldYielder(to_absolute_path(cfg.test_file), input_pipe=to_absolute_path(cfg.pipe_file))
    test_df = test_fy.get_df(inc_inputs=True, deprocess=False)
    validation_data = lgb.Dataset(test_df[train_features], label=test_df[target_feature], weight=test_df[weight_feature], reference=train_data)


    with mlflow.start_run():
        model = lgb.train(OmegaConf.to_object(cfg.model_param), train_data, valid_sets=[train_data, validation_data])
        y_proba_test = model.predict(test_df[train_features])
        y_pred_test = y_proba_test > 0.5
        loss = log_loss(test_df[target_feature], y_proba_test)
        acc = accuracy_score(test_df[target_feature], y_pred_test)
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

        fig, ax = plt.subplots()
        plt.hist(y_proba_test[test_df[target_feature]==0], alpha=0.4, bins=30, density=True, label='bkgr')
        plt.hist(y_proba_test[test_df[target_feature]==1], alpha=0.4, bins=30, density=True, label='sig')
        plt.legend()
        mlflow.log_figure(fig, "figure.png")

if __name__ == '__main__':
    main()
