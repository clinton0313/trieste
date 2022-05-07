#!bin/bash

#Backs up all csv files to a local directory outside of trieste

script_dir=$(dirname $(realpath $0))
backup_dir="$script_dir/../../benchmarking_backup"

mkdir $backup_dir
for FILE in $script_dir
do
    find $DIR -iname '*.csv' -exec cp -fv {} ${backup_dir} \;
done

echo "Files backed up:"
ls $backup_dir
