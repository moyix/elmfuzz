#!/bin/bash

rundir="$1"
prev_gen="$2"
next_gen="$3"
num_gens="$4"

# MODELS="codellama starcoder starcoder_diff"
MODELS="codellama"
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_RESET='\033[0m'
printf "$COLOR_GREEN"'============> %s: %6s -> %6s of %3d <============'"$COLOR_RESET"'\n' $rundir $prev_gen $next_gen $num_gens

# Get the top 10 generators from the last generation
mkdir -p "$rundir"/${next_gen}/{variants,seeds,logs}

# If this is the first generation, just use the seed
if [ "$prev_gen" == "initial" ]; then
    echo "First generation; using seed:" "$rundir"/initial/seeds/*.py
    cp "$rundir"/initial/seeds/*.py "$rundir"/${next_gen}/seeds/
    # Create the random gif directory
    mkdir -p /fastdata/randomgifs/${rundir}
else
    python analyze_cov.py "$rundir"/*/logs/coverage.json | sort -n | tail -n 10 | \
        while read cov gen model generator ; do
            echo "Selecting $generator from $gen/$model with $cov edges covered"
            cp "$rundir"/${gen}/variants/${model}/${generator}.py \
            "$rundir"/${next_gen}/seeds/${gen}_${model}_${generator}.py
        done
fi

# Generate the next generation. If this is the first generation, create 100 variants
# for each seed with each model. Otherwise, create 10 variants for each seed with each model.
if [ "$prev_gen" == "initial" ]; then
    NUM_VARIANTS=300
else
    NUM_VARIANTS=30
fi
echo "Generating next generation: ${NUM_VARIANTS} variants for each seed with each model"
# StarCoder
# for seed in "$rundir"/${prev_gen}/seeds/*.py ; do
#     echo "====================== starcoder: $seed ======================"
#     python genvariants_async.py --endpoint http://127.0.0.1:8192 -t 0.7 -m 2048 -n ${NUM_VARIANTS} \
#         -O "$rundir"/${next_gen}/variants/starcoder "$seed"
# done
# Code Llama
model=codellama
echo "====================== $model ======================"
expected_variants=$(ls "$rundir"/${next_gen}/seeds/*.py | wc -l)
expected_variants=$((expected_variants * NUM_VARIANTS))
python -u genvariants_parallel.py --endpoint http://127.0.0.1:8192 -s 9 -t 0.8 -m 2048 -n ${NUM_VARIANTS} \
    -O "$rundir"/${next_gen}/variants/"$model" -L "$rundir"/${next_gen}/logs/meta \
    "$rundir"/${next_gen}/seeds/*.py | \
    python -u drive_all.py -n 100 -s .gif \
        -L "$rundir"/${next_gen}/logs/outputgen_${model}.jsonl \
        -O /fastdata/randomgifs/${rundir}/${next_gen}/${model} \
        generate_random_gif \
        $expected_variants
# StarCoder_diff
# for seed in "$rundir"/${prev_gen}/seeds/*.py ; do
#     echo "====================== starcoder_diff: $seed ======================"
#     python genvariants_diff.py -c 'Improve the random gif generator' -t 0.7 -m 2048 -n ${NUM_VARIANTS} \
#         -O "$rundir"/${next_gen}/variants/starcoder_diff "$seed"
# done

# Collect the coverage of the generators
echo "Collecting coverage of the generators"
python getcov.py -O "$rundir"/${next_gen}/logs/coverage.json /fastdata/randomgifs/${rundir}/${next_gen}/

# Plot cumulative coverage so far
python analyze_cov.py -m $num_gens -c -p "$rundir"/*/logs/coverage.json
