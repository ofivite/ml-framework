cd mlruns/$1
for d in */; do
    cd -
    d="${d%/}"
    echo {$d}
    python predict.py --config-name for_evaluation.yaml year=2022 experiment_id=$1 run_id=$d
    python evaluate.py experiment_id=6 run_id=$d dataset=test
    python evaluate.py experiment_id=6 run_id=$d dataset=train
    cd mlruns/$1
done
cd -