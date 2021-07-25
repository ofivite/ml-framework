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

    # enable auto logging for mlflow
    mlflow.lightgbm.autolog(log_models=False) # models are logged separately for each fold

    # fetch feature/weight/target names
    train_features = cfg.cont_features + cfg.cat_features # features to be used in training
    weight_name = cfg.weight_name
    target_name = 'gen_target' # internal target name defined inside of lumin

    # prepare train/test data
    train_fy = FoldYielder(train_file, input_pipe=input_pipe_file)
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False, verbose=False, suppress_warn=True)
    train_df['w_cp'] = train_fy.get_column('w_cp')
    train_df['w_class_imbalance'] = train_fy.get_column('w_class_imbalance')
    train_df['plot_weight'] = train_fy.get_column('weight')
    train_df[cfg.logo_feature] = train_fy.get_column(cfg.logo_feature)
    train_df['fold_id'] = train_df[cfg.logo_feature] % cfg.n_splits
    #
    test_fy = FoldYielder(test_file, input_pipe=input_pipe_file)
    test_df = test_fy.get_df(inc_inputs=True, deprocess=False, verbose=False, suppress_warn=True)
    test_df['plot_weight'] = test_fy.get_column('weight')

    logo = LeaveOneGroupOut() # splitter into folds for training/validation
    with mlflow.start_run():
        for i_fold, (train_idx, validation_idx) in enumerate(logo.split(train_df, groups=train_df['fold_id'])):
            train_fold_df = train_df.iloc[train_idx]
            validation_fold_df = train_df.iloc[validation_idx]

            # check that `i_fold` is the same as `fold_id` corresponding to each`validation_idx` split
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

        if cfg.model_param.objective == 'binary':
            y_proba_test = model.predict(test_df[train_features])
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
            y_pred_class = np.argmax(y_proba, axis=-1)
            y_pred_class_proba = np.max(y_proba, axis=-1)
            df_pred = pd.DataFrame({'pred_class_proba': y_pred_class_proba, 'pred_class': y_pred_class,
                                    'true_class': test_df.gen_target, 'plot_weight': test_df['plot_weight']
                                    })

            # check that class id match in data and in training cfg
            class_ids = {int(class_id) for class_id in cfg.class_to_info}
            assert set(train_df.gen_target) == class_ids
            assert set(test_df.gen_target) == class_ids

            for class_id in cfg.class_to_info:
                class_name = cfg.class_to_info[class_id].name
                fig_density = plot_class_score(df_pred, class_id, cfg.class_to_info, how='density')
                fig_stacked = plot_class_score(df_pred, class_id, cfg.class_to_info, how='stacked', weight='plot_weight')
                mlflow.log_figure(fig_density, f"plots/density/multiclass_score_{class_name}.html")
                mlflow.log_figure(fig_stacked, f"plots/stacked/multiclass_score_{class_name}.html")

if __name__ == '__main__':
    main()
