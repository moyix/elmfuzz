#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_dir> <orig> <variant> [<variant> ...]" >&2
    exit 1
fi

output_dir=$1 ; shift
orig=$1 ; shift

mkdir -p ${output_dir}

orig_ansi=/tmp/$(basename ${orig}).ansi
pygmentize -P style=vs "${orig}" > ${orig_ansi}
for f in $@; do
    echo $f
    var_ansi=/tmp/$(basename ${f}).ansi
    pygmentize -P style=vs "${f}" > ${var_ansi}
    diff -B -a -U 10000 ${orig_ansi} ${var_ansi} | ansi2html -l -s osx-basic > ${output_dir}/$(basename ${f}).diff.html
    # Fix background color to white
    sed -i 's/background-color: #AAAAAA/background-color: #FFFFFF/g' ${output_dir}/$(basename ${f}).diff.html
    rm ${var_ansi}
done
rm ${orig_ansi}
