#!/usr/bin/env python3

import random
import requests
import argparse
import os

ENDPOINT = 'http://127.0.0.1:8192'

def model_info():
    """Get information about the model."""
    return requests.get(f'{ENDPOINT}/info').json()

def generate_completion(
        prompt,
        temperature=0.2,
        max_new_tokens=2048,
        repetition_penalty=1.1,
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
    return requests.post(f'{ENDPOINT}/generate', json=data).json()

def random_diff(text: str, msg: str) -> str:
    """Generate a prompt for StarCoder's diff format"""
    return f'<commit_before>{text}<commit_msg>{msg}<commit_after>'

def main():
    global ENDPOINT
    parser = argparse.ArgumentParser(
        description='Generate variants of a file using an LLM code diff model',
    )
    parser.add_argument('file', type=str)
    parser.add_argument('-n', '--num', type=int, default=1)
    parser.add_argument('-O', '--output', type=str, default='.')
    parser.add_argument('--endpoint', type=str, default=ENDPOINT)
    parser.add_argument('-c', '--commit-message', type=str, default='Numerous improvements')
    # Generation params
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=2048)
    parser.add_argument('-r', '--repetition-penalty', type=float, default=1.1)
    args = parser.parse_args()
    ENDPOINT = args.endpoint

    info = model_info()
    model = info['model_id']
    if model != 'bigcode/starcoder':
        parser.error("Diffs only supported for StarCoder")

    os.makedirs(args.output, exist_ok=True)

    for i in range(args.num):
        prompt = random_diff(open(args.file).read(), args.commit_message)
        res = generate_completion(
            prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        if 'generated_text' not in res:
            print(f"WARNING: no generated text in response: {res}")
            continue
        text = res['generated_text']
        # one of [length, eos_token, stop_sequence]
        finish_reason = res['details']['finish_reason']
        finish_reason = {
            'length': 'len',
            'eos_token': 'eos',
            'stop_sequence': 'stp',
        }[finish_reason]
        # Count lines
        gen_lines = text.count('\n')
        # filename and extension
        base, ext = os.path.splitext(args.file)
        out_file = f'{base}.var_{i:04}.diffmode.gen_{gen_lines:03}-fin_{finish_reason}{ext}'
        with open(os.path.join(args.output,out_file), 'w') as f:
            f.write(text)
        print(f'Wrote {out_file} to {args.output}')

if __name__ == '__main__':
    main()
