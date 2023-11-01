import io
import json
import os
import sys
import argparse
import importlib
from enum import Enum
import signal
import traceback
import resource
from typing import BinaryIO, Callable, Tuple
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout, redirect_stderr
import logging

from drive_log import set_loglevel
logger = logging.getLogger('root')

def exception_info(e: Exception, module_path: str):
    exception_type, exception_object, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    return {
        'class': f'{exception_type.__module__}.{exception_type.__name__}',
        'module': module_path,
        'filename': filename,
        'line': line_number,
        'exception': str(e),
        'traceback': traceback.format_tb(exception_traceback),
    }

# Context manager for timing out a function
class TimedExecution():
    def __init__(self, timeout):
        self.timeout = timeout
        self.timed_out = False
        self.old_handler = None

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.timeout)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_handler)
        return self.timed_out

    def _handle_timeout(self, signum, frame):
        self.timed_out = True
        raise TimeoutError(f"Timed out after {self.timeout} seconds")

# Context manager to run in a temporary directory
class TemporaryDirectoryContext():
    def __init__(self, *args, **kwargs):
        self.td = TemporaryDirectory(*args, **kwargs)
        self.old_cwd = None

    def __enter__(self):
        self.old_cwd = os.getcwd()
        self.td.__enter__()
        os.chdir(self.td.name)
        return self.td

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.old_cwd)
        self.td.__exit__(exc_type, exc_value, traceback)

# Context manager to limit RAM usage
class MemoryLimit():
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = resource.getrlimit(resource.RLIMIT_AS)
        # Only change the soft limit so that we can set it back
        resource.setrlimit(resource.RLIMIT_AS, (self.limit, self.old_limit[1]))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        resource.setrlimit(resource.RLIMIT_AS, self.old_limit)

# Context manager that combines the above
class Sandbox():
    def __init__(self, timeout, memory_limit):
        self.timeout = TimedExecution(timeout)
        self.memory_limit = MemoryLimit(memory_limit)
        self.tempdir = TemporaryDirectoryContext()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.capture_stdout = redirect_stdout(self.stdout)
        self.capture_stderr = redirect_stderr(self.stderr)

    def __enter__(self):
        self.timeout.__enter__()
        self.memory_limit.__enter__()
        self.tempdir.__enter__()
        self.capture_stdout.__enter__()
        self.capture_stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.timeout.__exit__(exc_type, exc_value, traceback)
        self.memory_limit.__exit__(exc_type, exc_value, traceback)
        self.tempdir.__exit__(exc_type, exc_value, traceback)
        self.capture_stdout.__exit__(exc_type, exc_value, traceback)
        self.capture_stderr.__exit__(exc_type, exc_value, traceback)

class SingleGenResult(str,Enum):
    Success = "Success"
    Timeout = "Timeout"
    TooBig  = "TooBig"
    Error   = "Error"

class TooBigException(Exception):
    pass

class SizeLimitedBinaryFile(io.BufferedWriter):
    def __init__(self, *args, max_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def write(self, b: bytes) -> int:
        new_position = self.tell() + len(b)
        if new_position > self.max_size:
            raise TooBigException(f"Writing would exceed the size limit of {self.max_size} bytes")
        return super().write(b)

def generate_one(
        i: int,
        function: Callable[[BinaryIO,BinaryIO],None],
        args: argparse.Namespace,
    ) -> Tuple[int, SingleGenResult, str, "dict[str,str]"]:
    # Function takes a file-like BytesIO object (/dev/urandom)
    # and a writable BytesIO file object (output file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    output_file = f'{args.output_prefix}_{i:08d}{args.suffix}'
    with open('/dev/urandom', 'rb') as rng, \
            open(output_file, 'wb') as out_base:
        out = SizeLimitedBinaryFile(out_base, max_size=args.size_limit)
        try:
            with Sandbox(args.timeout, args.max_mem) as s:
                function(rng, out)
            return (i, SingleGenResult.Success, 'Success', {
                'stdout': s.stdout.getvalue(),
                'stderr': s.stderr.getvalue(),
            })
        except TooBigException:
            return (
                i, SingleGenResult.TooBig,
                f"Output file {output_file} exceeded size limit of {args.size_limit} bytes",
                {
                    'stdout': s.stdout.getvalue(),
                    'stderr': s.stderr.getvalue(),
                },
            )
        except TimeoutError as e:
            return (i, SingleGenResult.Timeout, str(e), {
                'stdout': s.stdout.getvalue(),
                'stderr': s.stderr.getvalue(),
            })
        except Exception as e:
            return (i, SingleGenResult.Error,
                    f"Error running function {function.__name__}: {e}",
                    {
                        'stdout': s.stdout.getvalue(),
                        'stderr': s.stderr.getvalue(),
                    },
                )
        finally:
            out.close()

def result_stats(outcomes):
    counts = {}
    for _, result, _, _ in outcomes:
        counts[result.name] = counts.get(result.name, 0) + 1
    return counts

def get_function(module_path, function_name, args):
    try:
        full_module_path = os.path.abspath(module_path)
        # Don't use filename as module name, because it might not be a valid
        spec = importlib.util.spec_from_file_location(
            'generator_module',
            full_module_path,
        )
        module = importlib.util.module_from_spec(spec)
        # This needs to wrapped in the sandbox because modules can exec
        # code on load
        with Sandbox(args.timeout, args.max_mem) as s:
            spec.loader.exec_module(module)
            # Don't add the module to sys.modules, because I'm paranoid that if
            # this function is invoked twice in the same process it will have
            # some kind of caching issues.
            # sys.modules[module_path] = module
            function = getattr(module, function_name)
            return function
    except ModuleNotFoundError as e:
        logger.info(f"Could not find module {module_path}")
        return {
            'error': exception_info(e,module_path),
            'data': {
                'stats': { 'Error': args.num_iterations },
                'outcomes': [
                    (i, SingleGenResult.Error, "", {
                        'stdout': s.stdout.getvalue(),
                        'stderr': s.stderr.getvalue(),
                    })
                    for i in range(args.num_iterations)
                ],
            },
        }
    except Exception as e:
        logger.info(f"Error importing module {module_path}: {e}")
        return {
            'error': exception_info(e,module_path),
            'data': {
                'stats': { 'Error': args.num_iterations },
                'outcomes': [
                    (i, SingleGenResult.Error, "", {
                        'stdout': s.stdout.getvalue(),
                        'stderr': s.stderr.getvalue(),
                    })
                    for i in range(args.num_iterations)
                ],
            },
        }
def generate_all(module_path, function_name, args):
    function_or_error = get_function(module_path, function_name, args)
    if not callable(function_or_error):
        return function_or_error
    else:
        function = function_or_error

    outcomes = []
    try:
        with TimedExecution(args.total_timeout) as te:
            for i in range(args.num_iterations):
                outcome = generate_one(i, function, args)
                _, result, message, output = outcome
                logger.info(f"{i:04}: {result.name} - {message}: {output}")
                outcomes.append(
                    outcome
                )
    except TimeoutError as e:
        logger.info(f"Total timeout of {args.total_timeout} seconds exceeded")
        logger.info(f"Did {len(outcomes)} iterations")

    stats = result_stats(outcomes)
    return {
        'error': None,
        'data': {
            'stats': stats,
            'outcomes': outcomes,
        },
    }

def make_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'function', type=str,
        help='The function to run, in the format module_path.py:function_name',
    )
    parser.add_argument('-n', '--num-iterations', type=int, default=100)
    parser.add_argument(
        '-S', '--size-limit', type=int, default=50*1024*1024,
        help='Maximum size of the output file (in bytes)')
    parser.add_argument(
        '-o', '--output-prefix', type=str, default='output',
        help='Prefix for the output file name; the iteration number will be appended')
    parser.add_argument(
        '-s', '--suffix', type=str, default='.dat',
        help='Suffix for the output file name')
    parser.add_argument(
        '-t', '--timeout', type=int, default=10,
        help='Timeout for each iteration (in seconds)')
    parser.add_argument(
        '-T', '--total-timeout', type=int, default=300,
        help='Total timeout (in seconds)')
    parser.add_argument(
        '-M', '--max-mem', type=int, default=1024*1024*1024,
        help='Maximum memory usage (in bytes)')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def main():
    parser = make_parser('Run an input generator function in a loop')
    args = parser.parse_args()
    set_loglevel(logger, args)

    module_path, function_name = args.function.split(':')
    result = generate_all(module_path, function_name, args)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
