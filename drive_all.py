#!/usr/bin/env python3

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
import sys
from typing import BinaryIO
try:
    # Not until 3.11
    from hashlib import file_digest
except ImportError:
    def file_digest(f: BinaryIO):
        import hashlib
        BLOCKSIZE = 65536
        hasher = hashlib.sha256()
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
        return hasher

from drive_log import setup_custom_logger
logger = setup_custom_logger('root')

from tqdm import tqdm
from driver import generate_all, exception_info, make_parser

def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

def draw_success_rate(stats):
    COLOR_GREEN = '\033[92m'
    COLOR_RED = '\033[91m'
    COLOR_YELLOW = '\033[93m'
    COLOR_BLUE = '\033[94m'
    COLOR_END = '\033[0m'
    BOX = '▓'
    WIDTH = 80
    def bar(color, width):
        return f"{color}{BOX*width}{COLOR_END}"
    total = sum(stats.values())
    success = int(WIDTH * stats.get('Success', 0)   / total)
    too_big = int(WIDTH * stats.get('TooBig', 0)    / total)
    error   = int(WIDTH * stats.get('Error', 0)     / total)
    timeout = int(WIDTH * stats.get('Timeout', 0)   / total)

    # Calculate how much width is left after the outcomes are drawn
    used_width = success + too_big + error + timeout
    remaining_width = WIDTH - used_width

    # Add any remaining width to the largest outcome
    outcomes = [('Success', success), ('TooBig', too_big), ('Error', error), ('Timeout', timeout)]
    largest_outcome = max(outcomes, key=lambda x: x[1])[0]

    if largest_outcome == 'Success':
        success += remaining_width
    elif largest_outcome == 'TooBig':
        too_big += remaining_width
    elif largest_outcome == 'Error':
        error += remaining_width
    elif largest_outcome == 'Timeout':
        timeout += remaining_width

    drawn_bar = (bar(COLOR_GREEN, success) +
                 bar(COLOR_YELLOW, too_big) +
                 bar(COLOR_RED, error) +
                 bar(COLOR_BLUE, timeout))
    legend = (COLOR_GREEN + BOX + COLOR_END + " Success " +
              COLOR_YELLOW + BOX + COLOR_END + " TooBig " +
              COLOR_RED + BOX + COLOR_END + " Error " +
              COLOR_BLUE + BOX + COLOR_END + " Timeout")
    return drawn_bar + "    " + legend

def generate_stats(logfile):
    def add_stats(d1, d2):
        return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}
    # Track stats as we go and print them at the end
    running_stats = defaultdict(dict)
    with open(logfile) as f:
        original_args = json.loads(f.readline())['data']['args']
        for line in f:
            result = json.loads(line)
            data = result.get('data', {}) or {}
            if not data:
                # No data, so treat the whole batch as an error
                if 'infilled' in module_path:
                    running_stats['infilled'] = add_stats(
                        running_stats['infilled'],
                        {'Error': original_args['num_iterations']},
                    )
                elif 'complete' in module_path:
                    running_stats['complete'] = add_stats(
                        running_stats['complete'],
                        {'Error': original_args['num_iterations']},
                    )
                elif 'diffmode' in module_path:
                    running_stats['diffmode'] = add_stats(
                        running_stats['diffmode'],
                        {'Error': original_args['num_iterations']},
                    )
                continue
            module_path = result['data'].get('module_path')
            if 'infilled' in module_path and 'stats' in data:
                running_stats['infilled'] = add_stats(running_stats['infilled'], data['stats'])
            elif 'complete' in module_path and 'stats' in data:
                running_stats['complete'] = add_stats(running_stats['complete'], data['stats'])
            elif 'diffmode' in module_path and 'stats' in data:
                running_stats['diffmode'] = add_stats(running_stats['diffmode'], data['stats'])

    print(f"Stats:", file=sys.stderr)
    for k in sorted(running_stats.keys()):
        print(f"  {k}: {running_stats[k]}", file=sys.stderr)
    for k in sorted(running_stats.keys()):
        print(f"  {k}: {draw_success_rate(running_stats[k])}", file=sys.stderr)
    combined = {}
    for k in running_stats:
        combined = add_stats(combined, running_stats[k])
    print(f"  combined: {combined}", file=sys.stderr)
    total = sum(combined.values())
    success = combined.get('Success', 0)
    print(f"     total: {total} files attempted", file=sys.stderr)
    print(f"   success: {success} files generated", file=sys.stderr)
    if total != 0:
        print(f"  success%: {success/total*100:.2f}%", file=sys.stderr)

def generate_filestats(logfile):
    def count_unique_files(outdir):
        try:
            files = os.listdir(outdir)
        except FileNotFoundError:
            return 0
        unique_files = set([
            file_digest(open(os.path.join(outdir, f), 'rb')).hexdigest()
            for f in files
        ])
        return len(unique_files)
    def file_sizes(outdir):
        try:
            return [
                os.path.getsize(os.path.join(outdir, f))
                for f in os.listdir(outdir)
            ]
        except FileNotFoundError:
            return []
    def new_filestats():
        return {
            'file_sizes': {},
            'unique_hashes': {},
        }
    # Tracks file stats for each module, keyed by generation type
    file_stats = defaultdict(lambda: defaultdict(new_filestats))
    with open(logfile) as f:
        # Grab the original args to get the number of module paths
        # which is the total number of lines in the file
        original_args = json.loads(f.readline())['data']['args']
        total = len(original_args['module_paths'])
        for line in tqdm(f, total=total, desc="Collecting file stats"):
            result = json.loads(line)
            work_dir = result['data'].get('worker_dir')
            if work_dir is None: continue
            module_path = result['data'].get('module_path')
            if module_path is None: continue
            if 'infilled' in module_path:
                generation_type = 'infilled'
            elif 'complete' in module_path:
                generation_type = 'complete'
            elif 'diffmode' in module_path:
                generation_type = 'diffmode'
            else:
                continue
            module_path = os.path.splitext(os.path.basename(module_path))[0]
            file_stats[generation_type][module_path]['file_sizes'] = file_sizes(work_dir)
            file_stats[generation_type][module_path]['unique_hashes'] = count_unique_files(work_dir)

    print(f"File stats:", file=sys.stderr)
    computed_file_stats = defaultdict(dict)
    total_unique = 0
    single_unique = 0
    zero_unique = 0
    # Keep both average and n so we compute the combined average correctly
    average_file_size = []
    average_nonzero_file_size = []
    for generation_type in file_stats:
        print(f"  {generation_type}:", file=sys.stderr)
        gen_total_unique = sum([
            file_stats[generation_type][module_path]['unique_hashes']
            for module_path in file_stats[generation_type]
        ])
        total_unique += gen_total_unique
        computed_file_stats[generation_type]['total_unique'] = gen_total_unique
        print(f"    total unique: {gen_total_unique}", file=sys.stderr)
        # Number of generators with just one unique file
        gen_single_unique = len([
            module_path
            for module_path in file_stats[generation_type]
            if file_stats[generation_type][module_path]['unique_hashes'] == 1
        ])
        single_unique += gen_single_unique
        computed_file_stats[generation_type]['single_unique'] = gen_single_unique
        print(f"    single unique: {gen_single_unique}", file=sys.stderr)
        # Number of generators with zero unique files
        gen_zero_unique = len([
            module_path
            for module_path in file_stats[generation_type]
            if file_stats[generation_type][module_path]['unique_hashes'] == 0
        ])
        zero_unique += gen_zero_unique
        computed_file_stats[generation_type]['zero_unique'] = gen_zero_unique
        print(f"    zero unique: {gen_zero_unique}", file=sys.stderr)
        # Average file size
        gen_file_sizes = [
            file_stats[generation_type][module_path]['file_sizes']
            for module_path in file_stats[generation_type]
        ]
        # concatenate all the lists
        gen_file_sizes = sum(gen_file_sizes, [])
        gen_avg_file_size = sum(gen_file_sizes) / len(gen_file_sizes) if len(gen_file_sizes) > 0 else 0
        nonzero_file_sizes = [s for s in gen_file_sizes if s > 0]
        avg_nonzero_file_size = sum(nonzero_file_sizes) / len(nonzero_file_sizes) if len(nonzero_file_sizes) > 0 else 0
        average_file_size.append((gen_avg_file_size, len(gen_file_sizes)))
        average_nonzero_file_size.append((avg_nonzero_file_size, len(nonzero_file_sizes)))
        computed_file_stats[generation_type]['average_file_size'] = gen_avg_file_size
        computed_file_stats[generation_type]['average_nonzero_file_size'] = avg_nonzero_file_size
        print(f"    average file size: {gen_avg_file_size:.2f} bytes", file=sys.stderr)
        print(f"    average nonzero file size: {avg_nonzero_file_size:.2f} bytes", file=sys.stderr)
    # Combined stats
    print(f"  combined:", file=sys.stderr)
    computed_file_stats['combined']['total_unique'] = total_unique
    print(f"    total unique: {total_unique}", file=sys.stderr)
    computed_file_stats['combined']['single_unique'] = single_unique
    print(f"    single unique: {single_unique}", file=sys.stderr)
    computed_file_stats['combined']['zero_unique'] = zero_unique
    print(f"    zero unique: {zero_unique}", file=sys.stderr)
    # Compute the combined average file size
    total_file_size = sum([s*n for s,n in average_file_size])
    total_nonzero_file_size = sum([s*n for s,n in average_nonzero_file_size])
    total_files = sum([n for s,n in average_file_size])
    total_nonzero_files = sum([n for s,n in average_nonzero_file_size])
    combined_avg_file_size = total_file_size / total_files if total_files > 0 else 0
    combined_avg_nonzero_file_size = total_nonzero_file_size / total_nonzero_files if total_nonzero_files > 0 else 0
    computed_file_stats['combined']['average_file_size'] = combined_avg_file_size
    computed_file_stats['combined']['average_nonzero_file_size'] = combined_avg_nonzero_file_size
    print(f"    average file size: {combined_avg_file_size:.2f} bytes", file=sys.stderr)
    print(f"    average nonzero file size: {combined_avg_nonzero_file_size:.2f} bytes", file=sys.stderr)

    # Include the raw file stats in the output
    computed_file_stats['infilled']['raw'] = file_stats['infilled']
    computed_file_stats['complete']['raw'] = file_stats['complete']
    computed_file_stats['diffmode']['raw'] = file_stats['diffmode']

    # Save to a new JSON file based on the output log's name
    output_log = os.path.splitext(logfile)[0]
    output_log += '.filestats.json'
    with open(output_log, 'w') as f:
        json.dump(computed_file_stats, f, indent=2)
    print(f"Wrote file stats to {output_log}", file=sys.stderr)

class filestats_action(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        if namespace.logfile is None:
            parser.error('Must specify --logfile with --stats-only')
        generate_stats(namespace.logfile)
        generate_filestats(namespace.logfile)

        parser.exit()

def main():
    # Inherit the parser from driver.py
    parser = make_parser('Run an input generator function in a loop')
    # Remove the function argument so it doesn't interfere with our new positional args
    remove_argument(parser, 'function')
    parser.add_argument(
        'function_name', type=str,
        help='The function to run in each module',
    )
    parser.add_argument(
        'module_paths', type=str, nargs='+',
        help='The modules to run the function in',
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=None,
        help='Maximum number of jobs to run in parallel',
    )
    parser.add_argument('--raise-errors', action='store_true',
                        help="Don't catch exceptions in the main driver loop")
    parser.add_argument('-L', '--logfile', type=str, default=None,
                        help='Log file for JSON results')
    parser.add_argument('--stats-only', action=filestats_action)
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if args.logfile is not None:
        output_log = open(args.logfile, 'w')
    else:
        output_log = sys.stdout

    print(json.dumps({'error': None, 'data': {'args': args.__dict__}}), file=output_log)
    # default handler for JSON output to deal with Enums
    # Call generate_all on each module in args.module_paths in parallel
    with ProcessPoolExecutor(max_workers=args.jobs) as executor, \
        tqdm(total=len(args.module_paths)) as progress:
        futures_to_paths = {}
        for module_path in args.module_paths:
            # Make a copy of args; we need to update the output prefix so that
            # each module's output goes to a different place.
            module_args = args.__dict__.copy()
            module_name = os.path.basename(module_path)
            # Strip extension
            module_name = os.path.splitext(module_name)[0]
            prefix_dir = os.path.dirname(module_args['output_prefix'])
            prefix_name = os.path.basename(module_args['output_prefix'])
            worker_dir = os.path.join(prefix_dir, module_name)
            worker_prefix = os.path.join(worker_dir, prefix_name)
            # Don't need to create the directory here; generate_all will do it
            module_args['output_prefix'] = worker_prefix
            module_args = argparse.Namespace(**module_args)

            future = executor.submit(
                generate_all,
                module_path, args.function_name, module_args
            )
            future.add_done_callback(lambda _: progress.update())
            futures_to_paths[future] = (module_path, worker_dir)
        for future in as_completed(futures_to_paths):
            module_path, worker_dir = futures_to_paths[future]
            try:
                result = future.result()
                data = result.get('data', {}) or {}
                data['module_path'] = module_path
                data['worker_dir'] = worker_dir
                result['data'] = data
                print(json.dumps(result), file=output_log)
            except Exception as e:
                if args.raise_errors: raise
                print(json.dumps({
                    'error': exception_info(e, module_path),
                    'data': {
                        'module_path': module_path,
                        'worker_dir': worker_dir,
                    },
                }), file=output_log)
        progress.close()

    if output_log != sys.stdout:
        output_log.close()

    # Collect statistics if we have a log
    if args.logfile is None: return

    # Print the stats out to stderr now that we're done
    generate_stats(args.logfile)
    generate_filestats(args.logfile)

if __name__ == '__main__':
    main()
