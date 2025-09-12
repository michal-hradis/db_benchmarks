source ~/projects/2025-05-11_db_benchmarks/env/bin/activate

for D in /mnt/zfs1/data2/2025-05-11_db_benchmarks/katka_sxn_csv/chunks.vec /mnt/zfs1/data2/2025-05-11_db_benchmarks/katka_sxn_csv.full/chunks.vec
do
	python ./add_language_labels.py --source-dir $D --target-dir ${D}.lang
done 

