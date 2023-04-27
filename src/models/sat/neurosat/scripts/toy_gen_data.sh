for ty in "train" "test" ; do
    rm -rf data/$ty/sr5
    mkdir -p data/$ty/sr5
    for i in {1..2}; do
	rm -rf dimacs/$ty/sr5/grp$i
	mkdir -p dimacs/$ty/sr5/grp$i
	python3 python/gen_sr_dimacs.py dimacs/$ty/sr5/grp$i 10000 --min_n 3 --max_n 5
	python3 python/dimacs_to_data.py dimacs/$ty/sr5/grp$i data/$ty/sr5 1000
    done;
done;
