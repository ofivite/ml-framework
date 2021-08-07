import os
import yaml
import gc
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import ROOT as R
import uproot
from lumin.nn.data.fold_yielder import FoldYielder
import numpy as np
import pandas as pd
import mlflow

from utils.processing import fill_placeholders
from utils.inference import load_models, predict_folds

@hydra.main(config_path="configs/predict")
def main(cfg: DictConfig) -> None:
    # fill placeholders in the cfg parameters
    input_path = to_absolute_path(fill_placeholders(cfg.input_path, {'{year}': cfg.year}))
    output_path = to_absolute_path(fill_placeholders(cfg.output_path, {'{year}': cfg.year}))
    os.makedirs(output_path, exist_ok=True)

    run_folder = to_absolute_path(f'mlruns/{cfg.mlflow_experimentID}/{cfg.mlflow_runID}/')
    models, n_splits, xtrain_split_feature = load_models(run_folder)

    # extract names of training features from mlflow-stored model
    # note: not checking that the set of features is the same across models
    misc_features = OmegaConf.to_object(cfg.misc_features)
    train_features = []
    with open(to_absolute_path(f'{run_folder}/artifacts/model_0/MLmodel'), 'r') as f:
        model_cfg = yaml.safe_load(f)
    for s in model_cfg['signature']['inputs'].split('{')[1:]:
        train_features.append(s.split("\"")[3])

    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    with mlflow.start_run(run_id=cfg.mlflow_runID):
        # loop over input fold files
        for sample_name in cfg.sample_names:
            print(f'\n--> Predicting {sample_name}')
            print(f"        loading data set")
            input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})

            # extract DataFrame from fold file
            fy = FoldYielder(f'{input_path}/{input_filename}')
            df = fy.get_df(inc_inputs=True, deprocess=False, nan_to_num=False, verbose=False, suppress_warn=True)
            for f in misc_features: # add misc features
                df[f] = fy.get_column(f)
            df['fold_id'] = df[xtrain_split_feature] % n_splits

            # run cross-inference for folds
            pred_dict = predict_folds(df, train_features, misc_features, 'fold_id', models)

            if cfg.kind == 'for_datacards':
                # store predictions in RDataFrame and snapshot it into output ROOT file
                print(f"        storing to output file")
                output_filename = fill_placeholders(cfg.output_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})
                if os.path.exists(f'{output_path}/{output_filename}'):
                    os.system(f'rm {output_path}/{output_filename}')
                R_df = R.RDF.MakeNumpyDataFrame(pred_dict)
                R_df.Snapshot(cfg.output_tree_name, f'{output_path}/{output_filename}')
                del(df, R_df); gc.collect()
            elif cfg.kind == 'for_evaluation':
                print(f"        storing to output file & log to mlflow")
                df_pred = pd.DataFrame(pred_dict)
                df_pred.to_csv(f'{output_path}/{output_filename}.csv')
                mlflow.log_artifact(f'{output_path}/{output_filename}.csv', artifact_path='pred')
            else:
                raise Exception(f'Unknown kind for prediction: {cfg.kind}')

        mlflow.log_params({'run_pred_on': f'{input_path}/{cfg.input_filename_template}',
                           'saved_pred_to': f'{output_path}/{cfg.output_filename_template}'
        })

if __name__ == '__main__':
    main()
