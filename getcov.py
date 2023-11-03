#!/usr/bin/env python3

import argparse
from collections import defaultdict
import json
import subprocess
import tempfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os

AFL_COV = '/home/moyix/git/afl-2.52b/afl-showmap'
PROG = '/home/moyix/git/gifdec/gifread'

def afl_cov(input_file):
    with tempfile.NamedTemporaryFile() as f:
        cov_file = f.name
        cmd = [AFL_COV, '-q', '-o', cov_file, '-m', 'none', '--', PROG, input_file]
        subprocess.run(cmd)
        with open(cov_file, 'r') as f:
            return set(l.strip() for l in f)

def main():
    parser = argparse.ArgumentParser("Get coverage for generated inputs")
    parser.add_argument('gendir', help='Base directory for generated inputs, structure: gendir/[model]/[generator]/*.gif')
    parser.add_argument('-O', '--output', type=str, default='output.json')
    args = parser.parse_args()
    combined_cov = defaultdict(set)
    with ThreadPoolExecutor(max_workers=64) as executor:
        worklist = []
        for model in glob.glob(os.path.join(args.gendir, '*')):
            for generator in glob.glob(os.path.join(model, '*')):
                for f in glob.glob(os.path.join(generator, '*.gif')):
                    worklist.append((
                        os.path.basename(model),
                        os.path.basename(generator),
                        f
                    ))
        futures = {}
        progress = tqdm(total=len(worklist), desc='Coverage')
        for model, generator, f in worklist:
            future = executor.submit(afl_cov, f)
            futures[future] = (model, generator, f)
            future.add_done_callback(lambda _: progress.update())
        for future in as_completed(futures):
            model, generator, f = futures[future]
            cov = future.result()
            combined_cov[(model, generator)].update(cov)
        progress.close()
    for (model, generator), cov in combined_cov.items():
        print(f'{model:>20} {generator} {len(cov)}')
    cov_dict = {}
    for (model, generator), cov in combined_cov.items():
        if model not in cov_dict:
            cov_dict[model] = {}
        cov_dict[model][generator] = list(cov)
    with open(args.output, 'w') as f:
        json.dump(cov_dict, f)
if  __name__ == '__main__':
    main()
