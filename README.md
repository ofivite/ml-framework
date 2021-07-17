# ML framework for HTT analyses

## Environment setup
It is recommended to run the code or install complementary packages from within a dedicated [conda](https://www.anaconda.com) environment, which can be set up from a `conda.yaml` file.

1) Firstly, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, for Linux x86_64:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Following the instructions one will be asked to specify the directory where Miniconda will be installed. In case of using the framework at `naf`, **make sure to specify the path in nfs area**: e.g. `/nfs/dust/cms/user/{USERNAME}/miniconda3`. This is needed to ensure that sufficient space is available to download python packages. Also, type _yes_ once asked for initialisation of condaand if you don't want the conda base environment to be activated on startup,  execute `conda config --set auto_activate_base false`. Then, relaunch the shell/relogin for the installation to take effect.

2) After that, clone the repo and install the conda environment (note: this might take some time):
```bash
git clone https://github.com/yaourtpourtoi/ml-framework.git
cd ml-framework
conda env create -f conda.yaml
```

3) Once the conda environment is created, it can be activated/deactivated with:
```bash
ENV_NAME=htt-ml
conda activate ${ENV_NAME}
# conda deactivate
```

4) At this point, one might want to register this environment in Jupyter as a kernel. To do that, execute inside of the environment the following command:
```bash
ENV_NAME=htt-ml
ipython kernel install --user --name=${ENV_NAME}
```
The newly created kernel can now be activated in the Jupyter selection menu (top-right corner in the Jupyter tab)

**NB:** other initialisations of environment variables (especially those of $PATH) might negatively interfere with conda. One example would be the command `cmsenv` or sourcing of CMSSW-related initialisation scripts. If something doesn't work with packages inside the conda environment, check that there is no such conflict with other initialisation procedures (e.g., happening automatically in `~/.bashrc` or `~/.zshrc`).   

**NB:** For correct `plotly` rendering in JupyterLab, [check](https://plotly.com/python/troubleshooting/#jupyterlab-problems) that extensions are enabled in Extension Manager (located in the corresponding tab on the left panel of the JupyterLab window) and `jupyterlab-plotly` is displayed amongst them. 

## Data preprocessing
Within the framework, input ROOT files are preprocessed and skimmed prior to training. This is done with `preprocess.py` script, which combines the input set of files (also referred to as _nodes_) into a single pandas dataframe, then performs necessary transformations/additions, then splits the dataframe into the output nodes and stores each into `hdf5` foldfile using `df2foldfile` function of [lumin](https://lumin.readthedocs.io/en/stable/) library.

There are two paths in the preprocessing dataflow: to prepare the data set for training and for prediction. This is done by and the flag `for_training` indicates the path to be followed 
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
_P.S.:_ Oppositely to the manual installation, running `mlflow run` without `--no-conda` flag automatically creates a conda environment from `conda.yaml` cfg file and runs the code from there.

## Tracking results
Once the training is done, UI interface to inspect the logged results of `mlflow` can be launched with (`-p` specifies the port id):
```bash
REMOTE_PORT_ID=5000
mlflow ui -p ${REMOTE_PORT_ID}
```

In case of running on a remote machine and not being able to open a browser (as a GUI) there, one can listen to a remote server via ssh and interact with it on a local machine (e.g. personal laptop). The commands below will make an ssh tunnel and forward remote port to a local one:
```bash
REMOTE_PORT_ID=5000
LOCAL_PORT_ID=5010
SERVER=${USERNAME}@${HOSTNAME}
ssh -N -f -L localhost:${LOCAL_PORT_ID}:localhost:${REMOTE_PORT_ID} ${SERVER}
```

Then one can access `mlflow ui` locally by going to http://localhost:5010 in a browser (here, `5010` is a local port id taken from a code snippet example).
