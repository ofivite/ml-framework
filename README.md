# ML framework for HTT analyses

## Environment setup
It is recommended to run the code/install necessary packages from within a dedicated conda environment, which can be set up from a `conda.yaml` file. Firstly, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and after that install the conda environment with:
```bash
conda env create -f conda.yaml
```

Once the conda environment is created, it can be activated/deactivated with:
```bash
ENV_NAME=htt-ml
conda activate $ENV_NAME
# conda deactivate
```

At this point, one might want to register this environment in Jupyter as a kernel. To do that, execute inside of env the following command:
```bash
ENV_NAME=htt-ml
ipython kernel install --user --name=$ENV_NAME
```
The newly created kernel can now be activated in the Jupyter selection menu (top-right corner of Jupyter window)

_Note:_ Oppositely to this manual installation, running `mlflow run` without `--no-conda` flag will create a conda environment based on `conda.yaml` cfg file by default.

_Note:_ For correct `plotly` rendering in JupyterLab, [check](https://plotly.com/python/troubleshooting/#jupyterlab-problems) that extensions are enabled in Extension Manager (located in the corresponding tab on the left panel of the JupyterLab window) and `jupyterlab-plotly` is displayed amongst them.   

## Data preprocessing
To run skimming of ROOT files into hdf5 adjust `configs/preprocess_cfg.yaml` and run:
```bash
python preprocess_data.py # one can also pass here additional params/override existing ones, see hydra docs for details
```

## Model training
To track the model training [`mlflow`](https://mlflow.org/docs/latest/index.html) project has been set up, see its description in `MLproject` file. There is currently two entry points: _binary_ (binary classification problem) and _multi_ (multiclass classification problem), where each runs `python train.py` with necessary parameters from `configs/training.yaml` added/overriden. There is [`hydra`](https://hydra.cc/docs/intro) used under the hood to parse those parameters.  

To train the model create an experiment (unless already done) and run it with `mlflow` specifying:
*  a corresponding entry point (`-e multi`)
*  name of the experiment for the run to be assigned to (`--experiment-name test`)
*  `--no-conda` to avoid creating new conda environment
*  mlflow params with their values (`-P num_iterations=5`, and one can also pass multiple ones)
*  project directory (`.` - current)

```bash
mlflow experiments create -n test
mlflow run -e multi --experiment-name test -P num_iterations=5 --no-conda .
```

## Tracking results
Once the training is done, UI interface to inspect the logged results of `mlflow` can be launched with (`-p` specifies the port id):
```bash
PORT_ID=5000
mlflow ui -p ${PORT_ID}
```

In case of running on a remote machine and not being able to open a browser (as a GUI) there, one can listen to a remote server via ssh and interact with it on a local machine (e.g. personal laptop). The commands below (fill in {...} with the corresponding values) will make an ssh tunnel and forward remote port to a local one:
```bash
SERVER=${USERNAME}@naf-cms${MACHINE_ID}.desy.de
ssh -N -f -L localhost:${LOCAL_PORT_ID}:localhost:${REMOTE_PORT_ID} ${SERVER}
```

Then one can access `mlflow ui` locally by going to http://localhost:${LOCAL_PORT} in a browser.
