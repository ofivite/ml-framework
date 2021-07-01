import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

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

def plot_class_score(df, class_id, class_to_info, how='density'):
    if how=='density':
        hist_data = [df.query(f'pred_class == {class_id} and true_class == {i}')['pred_class_proba'] for i in class_to_info]
        class_labels = [class_to_info[i].name for i in class_to_info]
        class_colors = [f'rgba({class_to_info[i].color}, 1.)' if i==class_id # emphasize category class by increase in transparency
                                                              else f'rgba({class_to_info[i].color}, .2)'
                                                              for i in class_to_info]
        fig = ff.create_distplot(hist_data, class_labels, bin_size=1e-2, histnorm='probability', show_curve=True, show_rug=False, colors=class_colors)
        fig.update_layout(
            title_text=f'{class_to_info[class_id].name} category',
            autosize=False,
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            # paper_bgcolor="LightSteelBlue",
        )
        return fig
    elif how=='stacked':
        fig = px.histogram(df.query(f'pred_class == {class_id}'), x="pred_class_proba", y='gen_weight',
                   color="true_class",
                   marginal="box", # or violin, or rug
                   barmode='group',
                   histfunc='sum',
                   nbins=50,
                   color_discrete_map={i: f'rgba({class_to_info[i].color}, {class_to_info[i].alpha})' for i in class_to_info},
                   log_y=True)
        fig.update_layout(
            title_text=f'{class_to_info[class_id].name} category',
            barmode='stack',
            autosize=False,
            width=1000,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        #     paper_bgcolor="LightSteelBlue",
        )
        return fig
    else:
        raise ValueError(f'Unknown value of how={how}: should be either \"density\" or \"stacked\"')

@hydra.main(config_path="configs", config_name="training_cfg")
def main(cfg: DictConfig) -> None:
    # enable auto logging for mlflow
    mlflow.lightgbm.autolog()

    # feature names to be used in training
    train_features = cfg.cont_features + cfg.cat_features
    weight_feature = 'gen_weight' # internal (hardcoded) column name in lumin
    target_feature = 'gen_target' # internal (hardcoded) column name in lumin

    # prepare train/test data
    train_fy = FoldYielder(to_absolute_path(cfg.train_file), input_pipe=to_absolute_path(cfg.pipe_file))
    train_df = train_fy.get_df(inc_inputs=True, deprocess=False)
    train_data = lgb.Dataset(train_df[train_features], label=train_df[target_feature], weight=train_df[weight_feature])
    #
    test_fy = FoldYielder(to_absolute_path(cfg.test_file), input_pipe=to_absolute_path(cfg.pipe_file))
    test_df = test_fy.get_df(inc_inputs=True, deprocess=False)
    validation_data = lgb.Dataset(test_df[train_features], label=test_df[target_feature], weight=test_df[weight_feature], reference=train_data)

    # check that class id match in data and in training cfg
    class_ids = {int(class_id) for class_id in cfg.class_to_info}
    assert set(train_df.gen_target) == class_ids
    assert set(test_df.gen_target) == class_ids

    with mlflow.start_run():
        model = lgb.train(OmegaConf.to_object(cfg.model_param), train_data, valid_sets=[train_data, validation_data])
        y_proba_test = model.predict(test_df[train_features])
        if cfg.model_param.objective == 'binary':
            y_pred_test = y_proba_test > 0.5
            loss = log_loss(test_df[target_feature], y_proba_test)
            acc = accuracy_score(test_df[target_feature], y_pred_test)
            mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

            fig, ax = plt.subplots()
            plt.hist(y_proba_test[test_df[target_feature]==0], alpha=0.4, bins=30, density=True, label='bkgr')
            plt.hist(y_proba_test[test_df[target_feature]==1], alpha=0.4, bins=30, density=True, label='sig')
            plt.legend()
            mlflow.log_figure(fig, "binary_score.png")
        if cfg.model_param.objective == 'multiclass':
            y_proba = model.predict(test_df[train_features])
            y_pred_class = np.argmax(y_proba, axis=1)
            y_pred_class_proba = np.max(y_proba, axis=1)
            df_pred = pd.DataFrame({'pred_class_proba': y_pred_class_proba, 'pred_class': y_pred_class, 'true_class': test_df.gen_target})
            for class_id in cfg.class_to_info:
                class_name = cfg.class_to_info[class_id].name
                fig_density = plot_class_score(df_pred, class_id, cfg.class_to_info, how='density')
                fig_stacked = plot_class_score(df_pred, class_id, cfg.class_to_info, how='stacked')
                mlflow.log_figure(fig_density, f"plots/density/multiclass_score_{class_name}.html")
                mlflow.log_figure(fig_stacked, f"plots/stacked/multiclass_score_{class_name}.html")

if __name__ == '__main__':
    main()
