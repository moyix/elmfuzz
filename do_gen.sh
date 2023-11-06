#!/bin/bash

prev_gen="$1"
next_gen="$2"

# Get the top 10 generators from the last generation
mkdir -p runs/${next_gen}/{variants,seeds,logs}
python analyze_cov.py runs/${prev_gen}/logs/gengif_cov.json | sort -n | tail -n 10 | \
    while read cov model generator ; do
        echo "Selecting $generator from $model with $cov edges covered"
        cp runs/${prev_gen}/variants/${model}/${generator}.py \
           runs/${next_gen}/seeds/${model}_${generator}.py
    done

# Generate the next generation
echo "Generating next generation: 10 variants for each seed with each model"
# StarCoder
for seed in runs/${prev_gen}/seeds/*.py ; do
    echo "====================== starcoder: $seed ======================"
    python genvariants_async.py --endpoint http://127.0.0.1:8192 -m 1200 -n 10 \
        -O runs/${next_gen}/variants/starcoder "$seed"
done
# Code Llama
for seed in runs/${prev_gen}/seeds/*.py ; do
    echo "====================== codellama: $seed ======================"
    python genvariants_async.py --endpoint http://127.0.0.1:8193 -m 1200 -n 10 \
        -O runs/${next_gen}/variants/codellama "$seed"
done
# StarCoder_diff
for seed in runs/${prev_gen}/seeds/*.py ; do
    echo "====================== starcoder_diff: $seed ======================"
    python genvariants_diff.py -c 'Improve the random gif generator' -m 1200 -n 10 \
        -O runs/${next_gen}/variants/starcoder_diff "$seed"
done

# Use the generators to generate outputs
echo "Gnerating outputs: 100 random gifs with each generator"
for model in codellama starcoder starcoder_diff; do
    echo "============= model: ${model} ============="
    python drive_all.py -n 100 -s .gif \
        -L runs/${next_gen}/logs/gengif_${model}.jsonl \
        -O /fastdata/randomgifs/${next_gen}/${model} \
        generate_random_gif \
        runs/${next_gen}/variants/${model}/*.py
done

# Collect the coverage of the generators
echo "Collecting coverage of the generators"
python getcov.py -O runs/${next_gen}/logs/gengif_cov.json /fastdata/randomgifs/${next_gen}/
