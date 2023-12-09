#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 seed rundir num_gens"
    exit 1
fi

seed="$1"
rundir="$2"
num_gens="$3"

mkdir -p "$rundir"/initial/{variants,seeds,logs}
cp "$seed" "$rundir"/initial/seeds/
./do_gen.sh "$rundir" initial gen0 $num_gens
for i in $(seq 0 $num_gens); do
    ./do_gen.sh "$rundir" gen$i gen$[i+1] $num_gens
done
