EXPERIMENTID=9
RUNID=01a4af9f47b449e1905ec97fb7db2734
YEAR=2022

for d in $@; do
    python predict.py --config-name for_evaluation.yaml year=$YEAR experiment_id=$EXPERIMENTID run_id=$RUNID +cutoff=$d
done

cd mlruns/$EXPERIMENTID/$RUNID/artifacts/pred/

for file in ./*.csv; do
    cd -
    FILE=$(basename -- "$file")
    python evaluate.py experiment_id=$EXPERIMENTID run_id=$RUNID dataset=${FILE%.*}
    cd mlruns/$EXPERIMENTID/$RUNID/artifacts/pred/
done