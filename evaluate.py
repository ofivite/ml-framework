import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd
import mlflow
from utils.plotting import plot_class_score

@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    print('\n--> Loading predictions')
    run_folder = to_absolute_path(f'mlruns/{cfg.mlflow_experimentID}/{cfg.mlflow_runID}/')
    df_pred = pd.read_csv(f'{run_folder}/artifacts/pred/{cfg.dataset}')

    # check that class id match in data and in training cfg
    class_ids = {int(class_id) for class_id in cfg.class_to_info}
    assert set(df_pred.gen_target) == class_ids
    class_names = []

    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    with mlflow.start_run(experiment_id=cfg.mlflow_experimentID, run_id=cfg.mlflow_runID):
        # plot density scores in each category
        for class_id in cfg.class_to_info:
            class_name = cfg.class_to_info[class_id]['name']
            class_names.append(class_name)
            print(f'\n--> Plotting density for class ({class_name})')
            fig_density = plot_class_score(df_pred, class_id, cfg.class_to_info, how='density')
            mlflow.log_figure(fig_density, f'plots/{cfg.dataset}/density_{class_name}.html')
            # fig_stacked = plot_class_score(df_pred, class_id, cfg.class_to_info, how='stacked', weight='plot_weight')

if __name__ == '__main__':
    main()
