#!/usr/bin/env python3

import argparse
import json

def main():
    parser = argparse.ArgumentParser("Analyze coverage")
    parser.add_argument('covfile', help='Coverage file')
    args = parser.parse_args()
    with open(args.covfile, 'r') as f:
        cov = json.load(f)
    for model, generators in cov.items():
        for generator, cov in generators.items():
            print(f'{len(cov):3} {model:<14} {generator}')

if __name__ == '__main__':
    main()
