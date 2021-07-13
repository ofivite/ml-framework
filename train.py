import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

@hydra.main(config_path="configs", config_name="train_cfg")
def main(cfg: DictConfig) -> None:
    # enable auto logging for mlflow
    mlflow.lightgbm.autolog()

    # fetch feature/weight/target names
    train_features = cfg.cont_features + cfg.cat_features # features to be used in training
    weight_name = cfg.weight_name
    target_name = 'gen_target' # internal target name defined inside of lumin

    # prepare train/test data
    train_fy = FoldYielder(to_absolute_path(cfg.train_file), input_pipe=to_absolute_path(cfg.pipe_file))
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False)
    train_df['w_cp'] = train_fy.get_column('w_cp')
    train_df['w_class_imbalance'] = train_fy.get_column('w_class_imbalance')
    train_df['plot_weight'] = train_fy.get_column('weight')
    train_data = lgb.Dataset(train_df[train_features], label=train_df[target_name], weight=train_df[weight_name])
    #
    test_fy = FoldYielder(to_absolute_path(cfg.test_file), input_pipe=to_absolute_path(cfg.pipe_file))
    test_df = test_fy.get_df(inc_inputs=True, deprocess=False)
    test_df['w_cp'] = test_fy.get_column('w_cp')
    test_df['w_class_imbalance'] = test_fy.get_column('w_class_imbalance')
    test_df['plot_weight'] = test_fy.get_column('weight')
    validation_data = lgb.Dataset(test_df[train_features], label=test_df[target_name], weight=test_df[weight_name], reference=train_data)

    # check that class id match in data and in training cfg
    class_ids = {int(class_id) for class_id in cfg.class_to_info}
    assert set(train_df.gen_target) == class_ids
    assert set(test_df.gen_target) == class_ids

    with mlflow.start_run():
        model = lgb.train(OmegaConf.to_object(cfg.model_param), train_data, valid_sets=[train_data, validation_data])
        y_proba_test = model.predict(test_df[train_features])
        if cfg.model_param.objective == 'binary':
            y_pred_test = y_proba_test > 0.5
            loss = log_loss(test_df[target_name], y_proba_test)
            acc = accuracy_score(test_df[target_name], y_pred_test)
            mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

            fig, ax = plt.subplots()
            plt.hist(y_proba_test[test_df[target_name]==0], alpha=0.4, bins=30, density=True, label='bkgr')
            plt.hist(y_proba_test[test_df[target_name]==1], alpha=0.4, bins=30, density=True, label='sig')
            plt.legend()
            mlflow.log_figure(fig, "binary_score.png")
        if cfg.model_param.objective == 'multiclass':
            y_proba = model.predict(test_df[train_features])
            y_pred_class = np.argmax(y_proba, axis=1)
            y_pred_class_proba = np.max(y_proba, axis=1)
            df_pred = pd.DataFrame({'pred_class_proba': y_pred_class_proba, 'pred_class': y_pred_class,
                                    'true_class': test_df.gen_target, 'plot_weight': test_df['plot_weight']
                                    })
            for class_id in cfg.class_to_info:
                class_name = cfg.class_to_info[class_id].name
                fig_density = plot_class_score(df_pred, class_id, cfg.class_to_info, how='density')
                fig_stacked = plot_class_score(df_pred, class_id, cfg.class_to_info, how='stacked', weight='plot_weight')
                mlflow.log_figure(fig_density, f"plots/density/multiclass_score_{class_name}.html")
                mlflow.log_figure(fig_stacked, f"plots/stacked/multiclass_score_{class_name}.html")

if __name__ == '__main__':
    main()
