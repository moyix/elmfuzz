#!/bin/bash

# Be strict about failures
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 rundir"
    exit 1
fi

# This needs to be the first thing we do because elmconfig.py uses it
# to find the config file ($ELMFUZZ_RUNDIR/config.yaml)
export ELMFUZZ_RUNDIR="$1"
export ELMFUZZ_RUN_NAME=$(basename "$ELMFUZZ_RUNDIR")
seeds=$(./elmconfig.py get run.seeds)
num_gens=$(./elmconfig.py get run.num_generations)
# Generations are zero-indexed
last_gen=$((num_gens - 1))
genout_dir=$(./elmconfig.py get run.genoutput_dir -s GEN=. -s MODEL=.)
# normalize the path
genout_dir=$(realpath -m "$genout_dir")
# Check if we should remove the output dirs if they exist
should_clean=$(./elmconfig.py get run.clean)
if [ -d "$genout_dir" ]; then
    if [ "$should_clean" == "True" ]; then
        echo "Removing generated outputs in $genout_dir"
        rm -rf "$genout_dir"
    else
        echo "Generated output directory $genout_dir already exists; exiting."
        echo "Set run.clean to True to remove existing rundirs."
        exit 1
    fi
fi
# See if we have any gen*, initial, or stamps directories
for pat in "gen*" "initial" "stamps"; do
    if compgen -G "$ELMFUZZ_RUNDIR"/$pat > /dev/null; then
        if [ "$should_clean" == "True" ]; then
            echo "Removing existing rundir(s):" "$ELMFUZZ_RUNDIR"/$pat
            rm -rf "$ELMFUZZ_RUNDIR"/$pat
        else
            echo "Found existing rundir(s):" "$ELMFUZZ_RUNDIR"/$pat
            echo "Set run.clean to True to remove existing rundirs."
            exit 1
        fi
    fi
done

mkdir -p "$ELMFUZZ_RUNDIR"/initial/{variants,seeds,logs}
# Stamp dir tells us when a generation is fully finished
# In the future this will let us resume a run
mkdir -p "$ELMFUZZ_RUNDIR"/stamps
cp -v $seeds "$ELMFUZZ_RUNDIR"/initial/seeds/
./do_gen.sh initial gen0
for i in $(seq 0 $last_gen); do
    ./do_gen.sh gen$i gen$[i+1]
done
