#!/usr/bin/env python3

import random
import argparse
import os
import aiohttp
import asyncio
import requests

ENDPOINT = 'http://127.0.0.1:8192'

def model_info():
    """Get information about the model."""
    return requests.get(f'{ENDPOINT}/info').json()

async def generate_completion(
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
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{ENDPOINT}/generate', json=data) as resp:
            return await resp.json()

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

def random_completion(text: str) -> [str,str]:
    """Generate a completion of the text starting from a random line.
    Always include at least 1 line to avoid an empty prompt."""
    text_lines = text.split('\n')
    # Pick a random line number to cut at
    cut_line = random.randint(1, len(text_lines) - 1)
    prompt_text = '\n'.join(text_lines[:cut_line])
    real_completion = '\n'.join(text_lines[cut_line:])
    return prompt_text, real_completion

def random_fim(text: str) -> [str,str,str]:
    """Fill in the middle of the text with a random completion."""
    text_lines = text.split('\n')
    # Random start and end lines. Make sure we always have at least
    # one line in each section.
    start_line = random.randint(0, len(text_lines) - 2)
    end_line = random.randint(start_line + 1, len(text_lines) - 1)
    prefix_text = '\n'.join(text_lines[:start_line]) + '\n'
    suffix_text = '\n'.join(text_lines[end_line:])
    real_middle = '\n'.join(text_lines[start_line:end_line])
    return prefix_text, suffix_text, real_middle

async def generate_variant(i, generators, model, args):
    # Pick a random generator
    generator = random.choice(generators)
    if generator == 'infilled':
        prefix, suffix, orig = random_fim(open(args.file).read())
        prompt = infilling_prompt(prefix, suffix)
        stop = []
    else:
        prefix, orig = random_completion(open(args.file).read())
        suffix = ''
        prompt = prefix
        stop = ['\nif', '\nclass', '\nfor', '\nwhile']
    res = await generate_completion(
        prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop,
    )
    if 'generated_text' not in res:
        print(f"WARNING: no generated text in response: {res}")
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
    base = os.path.basename(args.file)
    base, ext = os.path.splitext(base)
    out_file = f'var_{i:04}.{generator}.pre_{plines:03}-orig_{olines:03}-gen_{gen_lines:03}-suf_{slines:03}-fin_{finish_reason}.base_{base}{ext}'
    with open(os.path.join(args.output,out_file), 'w') as f:
        f.write(prefix + text + suffix)
    print(f'Wrote {out_file} to {args.output}')

async def main():
    global ENDPOINT
    global infilling_prompt
    parser = argparse.ArgumentParser(
        description='Generate variants of a file using an LLM code model',
    )
    parser.add_argument('file', type=str)
    parser.add_argument('--no-completion', action='store_true')
    parser.add_argument('--no-fim', action='store_true')
    parser.add_argument('-n', '--num', type=int, default=1)
    parser.add_argument('-O', '--output', type=str, default='.')
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
    if args.no_completion and args.no_fim:
        parser.error(f'Nothing to do')

    os.makedirs(args.output, exist_ok=True)

    generators = []
    if not args.no_completion:
        generators += ['complete']
    if not args.no_fim:
        generators += ['infilled']

    for i in range(args.num):
        await generate_variant(i, generators, model, args)

if __name__ == '__main__':
    asyncio.run(main())
