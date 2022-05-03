#!bin/bash
#Runs every passed python file as argument in series. For example: bash run_benchmarking.sh file1.py file2.py file3.py
for var in "$@"
do
    echo "Running $var..."
    python $var
    wait
    echo "Completed $var"
done