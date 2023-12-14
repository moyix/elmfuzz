#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
import re
import plotext as plt

gen_re = re.compile(r'gen(\d+)')

def print_cov(covfiles):
    data = []
    for covfile in covfiles:
        gen = int(gen_re.search(covfile).group(1))
        with open(covfile, 'r') as f:
            cov = json.load(f)
        for model, generators in cov.items():
            for generator, cov in generators.items():
                data.append((gen, model, generator, len(cov)))
    return data

def cumulative_cov(covfiles):
    cov_by_gen = defaultdict(set)
    for covfile in covfiles:
        gen = int(gen_re.search(covfile).group(1))
        with open(covfile, 'r') as f:
            cov = json.load(f)
        for model, generators in cov.items():
            for generator, cov in generators.items():
                cov_by_gen[gen].update(cov)
    cumulative = set()
    data = []
    for gen, cov in sorted(cov_by_gen.items()):
        cumulative.update(cov)
        data.append((gen, len(cumulative)))
    return data

def main():
    parser = argparse.ArgumentParser("Analyze coverage")
    parser.add_argument('covfiles', help='Coverage file', nargs='+')
    parser.add_argument('-c', '--cumulative', help='Report cumulative coverage', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot coverage', action='store_true')
    parser.add_argument('-m', '--max-gen', help='Maximum generation for plotting', type=int, default=None)
    args = parser.parse_args()

    rundir = args.covfiles[0].split('/')[0]
    if args.plot:
        # Don't fill the whole terminal
        width, height = plt.ts()
        plt.plotsize(width // 2, height // 2)
        if args.max_gen is not None:
            plt.xlim(0, args.max_gen)

    if not args.cumulative:
        data = print_cov(args.covfiles)
        if args.plot:
            plt.scatter([x[0] for x in data], [x[3] for x in data])
            plt.title(f'Variant coverage by generation, {rundir}')
            plt.xlabel('Generation')
            plt.ylabel('Edges')
            plt.show()
        else:
            for gen, model, generator, cov in data:
                gen_str = f'gen{gen}'
                print(f'{cov:3} {gen_str:<5} {model:<14} {generator}')
    else:
        data = cumulative_cov(args.covfiles)
        if args.plot:
            plt.plot([x[0] for x in data], [x[1] for x in data])
            plt.title(f'Cumulative coverage by generation, {rundir}')
            plt.xlabel('Generation')
            plt.ylabel('Edges')
            plt.show()
        else:
            for gen, cumulative in data:
                gen_str = f'gen{gen}'
                print(f'{gen_str:<5} {cumulative:3}')

if __name__ == '__main__':
    main()
