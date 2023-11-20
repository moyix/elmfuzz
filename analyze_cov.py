#!/usr/bin/env python3

import argparse
import json
import os
import re

def main():
    gen_re = re.compile(r'(gen\d+)')
    parser = argparse.ArgumentParser("Analyze coverage")
    parser.add_argument('covfiles', help='Coverage file', nargs='+')
    args = parser.parse_args()
    for covfile in args.covfiles:
        gen = gen_re.search(covfile).group(1)
        with open(covfile, 'r') as f:
            cov = json.load(f)
        for model, generators in cov.items():
            for generator, cov in generators.items():
                print(f'{len(cov):3} {gen:<5} {model:<14} {generator}')

if __name__ == '__main__':
    main()
