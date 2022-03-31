import os
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from utils.plotting import plot_class_score, plot_curves

@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    print('\n--> Loading predictions')
    run_folder = to_absolute_path(f'mlruns/{cfg.experiment_id}/{cfg.run_id}/')
    df_pred = pd.read_csv(f'{run_folder}/artifacts/pred/{cfg.dataset}.csv')

    # check that class id match in data and in training cfg
    class_ids = {int(class_id) for class_id in cfg.class_to_info}
    assert set(df_pred['target']) == class_ids
    class_names = []

    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id):
        # plot density scores in each category
        for class_id in cfg.class_to_info:
            class_name = cfg.class_to_info[class_id]['name']
            class_names.append(class_name)

            print(f'\n--> Plotting density for class ({class_name})')
            fig_density_name = f'density_{class_name}.pdf'
            fig_density = plot_class_score(df_pred, class_id, cfg.class_to_info, how='density')
            fig_density.write_image(fig_density_name)
            mlflow.log_figure(fig_density, f'plots/{cfg.dataset}/density_{class_name}.html')
            mlflow.log_artifact(fig_density_name, f'plots/{cfg.dataset}/pdf')
            os.remove(fig_density_name)
            # fig_stacked = plot_class_score(df_pred, class_id, cfg.class_to_info, how='stacked', weight='plot_weight')

        # make confusion matrix
        print(f'\n--> Producing confusion matrix')
        for confusion_norm in ['true', 'pred']:
            cm = confusion_matrix(df_pred['target'], df_pred['pred_class'], normalize=confusion_norm, sample_weight=df_pred['w_class_imbalance'])
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            for class_id in cfg.class_to_info:
                mlflow.log_metric(f'cm_{class_id}{class_id}_{confusion_norm} / {cfg.dataset}', cm[class_id,class_id])

            fig, ax = plt.subplots(figsize=(10, 9))
            disp.plot(cmap='Blues', ax=ax)
            cm_name = f'confusion_matrix_{confusion_norm}.pdf'
            ax.set_title(f'Confusion matrix: class balanced, normalize={confusion_norm}')
            fig.savefig(cm_name)
            mlflow.log_artifact(cm_name, f'plots/{cfg.dataset}/pdf')
            os.remove(cm_name)

        # plot ROC and precision-recall curves for each class
        print(f'\n--> Plotting ROC & PR curves')
        for curve_name, curve_data in plot_curves(df_pred, cfg.class_to_info).items():
            curve_data['figure'].write_image(f'{curve_name}_curve.pdf')
            mlflow.log_figure(curve_data['figure'], f'plots/{cfg.dataset}/{curve_name}_curve.html')
            for class_name in class_names:
                for metric_key in curve_data:
                    if ('auc' in metric_key or 'average_prec' in metric_key):
                        mlflow.log_metric(f'{curve_name}_{metric_key} / {cfg.dataset}', curve_data[metric_key])
            mlflow.log_artifact(f'{curve_name}_curve.pdf', f'plots/{cfg.dataset}/pdf')
            os.remove(f'{curve_name}_curve.pdf')
    print()

if __name__ == '__main__':
    main()
