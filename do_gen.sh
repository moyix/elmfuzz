#!/bin/bash

# Be strict about failures
set -euo pipefail

prev_gen="$1"
next_gen="$2"
num_gens=$(./elmconfig.py get run.num_generations)

# MODELS="codellama starcoder starcoder_diff"
MODELS=$(./elmconfig.py get model.names)
NUM_VARIANTS=$(./elmconfig.py get cli.genvariants_parallel.num_variants)
LOGDIR=$(./elmconfig.py get run.logdir -s GEN=${next_gen})

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_RESET='\033[0m'
printf "$COLOR_GREEN"'============> %s: %6s -> %6s of %3d <============'"$COLOR_RESET"'\n' $ELMFUZZ_RUN_NAME $prev_gen $next_gen $num_gens
echo "Running generation $next_gen using $MODELS with $NUM_VARIANTS variants per seed"

# Create the next generation directory
mkdir -p "$ELMFUZZ_RUNDIR"/${next_gen}/{variants,seeds,logs}

# Select the seeds for the next generation based on coverage
# If this is the first generation, just use the seed
if [ "$prev_gen" == "initial" ]; then
    echo "First generation; using seed(s):" "$ELMFUZZ_RUNDIR"/initial/seeds/*.py
    cp "$ELMFUZZ_RUNDIR"/initial/seeds/*.py "$ELMFUZZ_RUNDIR"/${next_gen}/seeds/
else
    # Selection
    selection_strategy=$(./elmconfig.py get run.selection_strategy)
    # If strategy is elites, select best coverage across all generations
    # If it's best_of_generation, select best coverage from the previous generation
    # Hopefully eventually we will also have MAP-Elites
    if [ "$selection_strategy" == "elites" ]; then
        echo "$selection_strategy: Selecting best seeds from all generations"
        cov_files=("$ELMFUZZ_RUNDIR"/*/logs/coverage.json)
    elif [ "$selection_strategy" == "best_of_generation" ]; then
        echo "$selection_strategy: Selecting best seeds from previous generation"
        cov_files=("$ELMFUZZ_RUNDIR"/${prev_gen}/logs/coverage.json)
    else
        echo "Unknown selection strategy $selection_strategy; exiting"
        exit 1
    fi
    python analyze_cov.py "${cov_files[@]}" | sort -n | tail -n 10 | \
        while read cov gen model generator ; do
            echo "Selecting $generator from $gen/$model with $cov edges covered"
            cp "$ELMFUZZ_RUNDIR"/${gen}/variants/${model}/${generator}.py \
               "$ELMFUZZ_RUNDIR"/${next_gen}/seeds/${gen}_${model}_${generator}.py
        done
fi

# Generate the next generation. If this is the first generation, create 10xNUM_VARIANTS variants
# for each seed with each model. Otherwise, create NUM_VARIANTS variants for each seed with each model.
if [ "$prev_gen" == "initial" ]; then
    NUM_VARIANTS=$((NUM_VARIANTS * 10))
    VARIANT_ARGS="-n ${NUM_VARIANTS}"
else
    VARIANT_ARGS=""
fi
echo "Generating next generation: ${NUM_VARIANTS} variants for each seed with each model"
for model_name in $MODELS ; do
    MODEL=$(basename "$model_name")
    GVLOG="${LOGDIR}/meta"
    GOLOG="${LOGDIR}/outputgen_${MODEL}.jsonl"
    GVOUT=$(./elmconfig.py get run.genvariant_dir -s MODEL=${MODEL} -s GEN=${next_gen})
    GOOUT=$(./elmconfig.py get run.genoutput_dir -s MODEL=${MODEL} -s GEN=${next_gen})
    echo "====================== $model_name ======================"
    # TODO: have genvariants_parallel.py do all the models at once
    # Will have to add in model-specific args to the config and merge
    # in the starcoder_diff script
    python genvariants_parallel.py $VARIANT_ARGS \
        -M "${model_name}" -O "$GVOUT" -L "$GVLOG" \
        "$ELMFUZZ_RUNDIR"/${next_gen}/seeds/*.py | \
        python genoutputs.py -L "${GOLOG}" -O "${GOOUT}"
done

# Collect the coverage of the generators
echo "Collecting coverage of the generators"
all_models_genout_dir=$(realpath -m "$GOOUT"/..)
python getcov.py -O "${LOGDIR}/coverage.json" "$all_models_genout_dir"

# Plot cumulative coverage so far
python analyze_cov.py -m $num_gens -c -p "$ELMFUZZ_RUNDIR"/*/logs/coverage.json

# Create a stamp file to indicate that this generation is finished
touch "$ELMFUZZ_RUNDIR"/stamps/${next_gen}.stamp
