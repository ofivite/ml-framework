EXPERIMENTID=9
RUNID=01a4af9f47b449e1905ec97fb7db2734

for d in $@; do
    sed -e "s;%CUT%;$d;g" configs/predict/for_evaluation.txt > configs/predict/for_evaluation.yaml
    python predict.py --config-name for_evaluation.yaml year=2022 experiment_id=$EXPERIMENTID run_id=$RUNID
done

cd mlruns/$EXPERIMENTID/$RUNID/artifacts/pred/

for file in ./*.csv; do
    cd -
    FILE=$(basename -- "$file")
    python evaluate.py experiment_id=$EXPERIMENTID run_id=$RUNID dataset=${FILE%.*}
    cd mlruns/$EXPERIMENTID/$RUNID/artifacts/pred/
done