#!/usr/bin/env python3

import json
import random
import argparse
import os
from tqdm import tqdm
import requests
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT = 'http://127.0.0.1:8192'

def model_info():
    """Get information about the model."""
    return requests.get(f'{ENDPOINT}/info').json()

def generate_completion(
        prompt,
        temperature=0.2,
        max_new_tokens=1200,
        repetition_penalty=1.1,
        stop=None,
):
    """Generate a completion of the prompt."""
    data = {
        'inputs': prompt,
        'parameters': {
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'repetition_penalty': repetition_penalty,
             "details": True, # So we get the finish_reason
        },
    }
    if stop is not None:
        data['parameters']['stop'] = stop
    return requests.post(f'{ENDPOINT}/generate', json=data).json()

def infilling_prompt_llama(
    pre: str,
    suf: str,
) -> str:
    """
    Format an infilling problem for Code Llama.
    If `suffix_first` is set, format in suffix-prefix-middle format.
    """
    return f'<PRE> {pre} <SUF>{suf} <MID>'

def infilling_prompt_starcoder(
    pre: str,
    suf: str,
) -> str:
    """
    Format an infilling problem for StarCoder
    If `suffix_first` is set, format in suffix-prefix-middle format.
    """
    return f'<fim_prefix>{pre}<fim_suffix>{suf}<fim_middle>'

infilling_prompt = None

def random_completion(text: str, start_line: int = 1) -> [str,str]:
    """Generate a completion of the text starting from a random line.
    Always include at least 1 line to avoid an empty prompt."""
    text_lines = text.split('\n')
    # Pick a random line number to cut at
    cut_line = random.randint(start_line + 1, len(text_lines) - 1)
    prompt_text = '\n'.join(text_lines[:cut_line])
    real_completion = '\n'.join(text_lines[cut_line:])
    return prompt_text, real_completion

def random_fim(text: str, start_line: int = 1) -> [str,str,str]:
    """Fill in the middle of the text with a random completion."""
    text_lines = text.split('\n')
    # Random start and end lines. Make sure we always have at least
    # one line in each section.
    fim_start_line = random.randint(start_line + 1, len(text_lines) - 2)
    fim_end_line = random.randint(fim_start_line + 1, len(text_lines) - 1)
    prefix_text = '\n'.join(text_lines[:fim_start_line]) + '\n'
    suffix_text = '\n'.join(text_lines[fim_end_line:])
    real_middle = '\n'.join(text_lines[fim_start_line:fim_end_line])
    return prefix_text, suffix_text, real_middle

def random_crossover(text1: str, text2: str, start_line: int = 1) -> [str,str]:
    """Generate a splice of two texts."""
    text_lines1 = text1.split('\n')
    text_lines2 = text2.split('\n')
    cut_line1 = random.randint(start_line + 1, len(text_lines1) - 1)
    # Cut line in file2.
    cut_line2 = random.randint(start_line + 1, len(text_lines2) - 1)
    prefix = '\n'.join(text_lines1[:cut_line1])
    suffix = '\n'.join(text_lines2[cut_line2:])
    return prefix, suffix

def new_base(filename: str) -> str:
    # filename and extension
    base = os.path.basename(filename)
    base, ext = os.path.splitext(base)
    # Get the first occurrence (if any) of ".base_"
    first = base.find('.base_')
    if first == -1:
        return base, ext
    else:
        base = base[:first]
        return base, ext

def generate_variant(i, generators, model, filename, args):
    # Pick a random generator
    generator = random.choice(generators)
    if generator == 'infilled':
        prefix, suffix, orig = random_fim(open(filename).read(), args.start_line)
        prompt = infilling_prompt(prefix, suffix)
        stop = []
    elif generator == 'lmsplice':
        other_files = [f for f in args.files if f != filename]
        if other_files:
            filename2 = random.choice(other_files)
        else:
            filename2 = filename
        prefix, suffix = random_crossover(open(filename).read(), open(filename2).read(), args.start_line)
        orig = ''
        prompt = infilling_prompt(prefix, suffix)
        stop = []
    else:
        prefix, orig = random_completion(open(filename).read(), args.start_line)
        suffix = ''
        prompt = prefix
        stop = ['\nif', '\nclass', '\nfor', '\nwhile']
    res = generate_completion(
        prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop,
    )
    if 'generated_text' not in res:
        # print(f"WARNING: no generated text in response: {res}")
        return None
    text = res['generated_text']
    if 'codellama' in model:
        # CodeLlama tokenizer decoding seems slightly broken in TGI,
        # so we need to remove the ' <EOT>' token manually, and trim the
        # stop sequences.
        text = text.replace(' <EOT>', '')
        for stop_seq in stop:
            if text.endswith(stop_seq):
                text = text[:-len(stop_seq)]
    # one of [length, eos_token, stop_sequence]
    finish_reason = res['details']['finish_reason']
    finish_reason = {
        'length': 'len',
        'eos_token': 'eos',
        'stop_sequence': 'stp',
    }[finish_reason]
    # Count lines
    plines = prefix.count('\n')
    slines = suffix.count('\n')
    olines = orig.count('\n')
    gen_lines = text.count('\n')
    # filename and extension
    base, ext = new_base(filename)
    if generator == 'lmsplice':
        base2, _ = new_base(filename2)
    else:
        base2 = base
    meta = {
        'model': model,
        'prompt': prompt,
        'generator': generator,
        'prompt_lines': plines,
        'orig_lines': olines,
        'gen_lines': gen_lines,
        'suffix_lines': slines,
        'finish_reason': finish_reason,
        'base': [base] + ([base2] if generator == 'lmsplice' else []),
        'response': res,
    }
    out_file = f'var_{i:04}.{generator}{ext}'
    out_path = os.path.join(args.output,out_file)
    with open(out_path, 'w') as f:
        f.write(prefix)
        f.write(text)
        f.write(suffix)
    # Write metadata to logdir
    with open(os.path.join(args.logdir, out_file + '.json'), 'w') as f:
        f.write(json.dumps(meta))
    # tqdm.write(f'Wrote {out_file} to {args.output}')
    return out_path

def main():
    global ENDPOINT
    global infilling_prompt
    parser = argparse.ArgumentParser(
        description='Generate variants of a file using an LLM code model',
    )
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--no-completion', action='store_true')
    parser.add_argument('--no-fim', action='store_true')
    parser.add_argument('--no-splice', action='store_true')
    parser.add_argument('-n', '--num', type=int, default=1)
    parser.add_argument('-O', '--output', type=str, default='.')
    parser.add_argument('-L', '--logdir', type=str, default='logs')
    parser.add_argument('-s', '--start-line', type=int, default=0,
                        help='When making random cuts, always start at this line')
    parser.add_argument('-j', '--jobs', type=int, default=16)
    parser.add_argument('--endpoint', type=str, default=ENDPOINT)
    # Generation params
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-r', '--repetition-penalty', type=float, default=1.1)
    args = parser.parse_args()
    ENDPOINT = args.endpoint

    info = model_info()
    model = info['model_id']
    if model == 'bigcode/starcoder':
        infilling_prompt = infilling_prompt_starcoder
    elif model in ('codellama/CodeLlama-13b-hf',
                   'codellama/CodeLlama-7b-hf'):
        infilling_prompt = infilling_prompt_llama
    else:
        infilling_prompt = None

    if infilling_prompt is None and not args.no_fim:
        parser.error(f'Model {model} does not support FIM')
    if args.no_completion and args.no_fim and args.no_splice:
        parser.error(f'Nothing to do')

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    generators = []
    if not args.no_completion:
        generators += ['complete']
    if not args.no_fim:
        generators += ['infilled']
    if not args.no_splice:
        generators += ['lmsplice']

    worklist = []
    i = 0
    for _ in range(args.num):
        for filename in args.files:
            worklist.append((i, filename))
            i += 1
    # pbar = tqdm(total=len(worklist), desc='Generating', unit='variant')
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = []
        for i, filename in worklist:
            future = executor.submit(generate_variant, i, generators, model, filename, args)
            # future.add_done_callback(lambda _: pbar.update())
            futures.append(future)
        for future in as_completed(futures):
            print(future.result(), flush=True)
    # pbar.close()

if __name__ == '__main__':
    main()
