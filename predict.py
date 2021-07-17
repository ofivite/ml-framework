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
from mlflow.pyfunc import load_model

from utils.processing import fill_placeholders

@hydra.main(config_path="configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    # fill placeholders in the cfg parameters
    input_path = to_absolute_path(fill_placeholders(cfg.input_path, {'{year}': cfg.year}))
    output_path = to_absolute_path(fill_placeholders(cfg.output_path, {'{year}': cfg.year}))
    os.makedirs(output_path, exist_ok=True)

    model_path = to_absolute_path(f'mlruns/{cfg.mlflow_experimentID}/{cfg.mlflow_runID}/artifacts/model')
    input_pipe = to_absolute_path(fill_placeholders(cfg.input_pipe, {'{year}': cfg.year}))

    # extract names of training features from mlflow-stored model
    misc_features = OmegaConf.to_object(cfg.misc_features)
    train_features = []
    with open(to_absolute_path(f'{model_path}/MLmodel'), 'r') as f:
        model_cfg = yaml.safe_load(f)
    for s in model_cfg['signature']['inputs'].split('{')[1:]:
        train_features.append(s.split("\"")[3])

    # loop over input fold files
    for sample_name in cfg.sample_names:
        print(f'--> {sample_name}')
        print(f"        loading ...")
        input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})

        # extract DataFrame from fold file
        fy = FoldYielder(f'{input_path}/{input_filename}', input_pipe=input_pipe)
        df = fy.get_df(inc_inputs=True, deprocess=False, verbose=False, suppress_warn=True)
        for f in misc_features: # add misc features
            df[f] = fy.get_column(f)

        # load mlflow logged model
        model = load_model(model_path)

        # make predictions
        print(f"        predicting ...")
        y_proba = model.predict(df[train_features])
        y_pred_class = np.argmax(y_proba, axis=1).astype(np.int32)
        y_pred_class_proba = np.max(y_proba, axis=1).astype(np.float32)

        # store predictions in RDataFrame and snapshot it into output ROOT file
        print(f"        storing to output file ...")
        output_filename = fill_placeholders(cfg.output_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})
        if os.path.exists(f'{output_path}/{output_filename}'):
            os.system(f'rm {output_path}/{output_filename}')
        R_df = R.RDF.MakeNumpyDataFrame({'pred_class': y_pred_class,
                                         'pred_class_proba': y_pred_class_proba,
                                          **{misc_feature: df[misc_feature].to_numpy() for misc_feature in misc_features}
                                         })
        R_df.Snapshot(cfg.output_tree_name, f'{output_path}/{output_filename}')
        del(df, R_df); gc.collect()
if __name__ == '__main__':
    main()
