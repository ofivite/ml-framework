# htt-ml-framework

To run skimming of ROOT files into hdf5 adjust `configs/preprocess_cfg.yaml` and run:
```bash
python preprocess_data.py # you can also pass here additional params/override existing ones, see hydra docs for details
```

To track the model training `mlflow` project has been set up, see its description in `MLproject` file. There is currently only one entry point `main`, which simply runs `python train.py`. This by default uses the params from `configs/training_cfg.yaml` as parsed by `hydra`, so adjust them accordingly.  To train the model create an experiment (unless already done) and run it with `mlflow` as (`--no-conda` to avoid creating new conda environment):
```bash
mlflow experiments create -n test # create an experiment "test"
mlflow run -e main --experiment-name test --no-conda . # run and log the code within the "test" experiment
```

Once done, UI interface to inspect the logged results of `mlflow` can be launched with (`-p` specifies the port id): 
```bash
mlflow ui -p 5000
```

In case of running on a remote machine and not being able to open a browser (as a GUI) there, one can listen to a remote server via an ssh tunnel and interact with it on a local machine (e.g. laptop):
```bash
SERVER=${USERNAME}@naf-cms${MACHINE_ID}.desy.de # remote machine
ssh -N -f -L localhost:${REMOTE_PORT}:localhost:${LOCAL_PORT} $SERVER # forwards REMOTE_PORT from SERVER to LOCAL_PORT and listens to it
```

Then one can access `mlflow ui` locally by going to http://localhost:${LOCAL_PORT} in a browser.
