from collections import namedtuple
import io
import json
import os
import shutil
import sys
import argparse
import importlib.util
from enum import Enum
import signal
import time
import traceback
import resource
from typing import BinaryIO, Callable, List, NamedTuple, Tuple, Union
from tempfile import TemporaryDirectory
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from drive_log import set_loglevel
logger = logging.getLogger('root')

class ExceptionInfo(NamedTuple):
    exception_class: str
    exception_message: str
    module_path: str
    filename: str
    line: int
    traceback: List[str]

    @classmethod
    def from_exception(cls, e: Exception, module_path: str):
        return cls(
            exception_class = f'{e.__class__.__module__}.{e.__class__.__name__}',
            exception_message = str(e),
            module_path = module_path,
            filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else None,
            line = e.__traceback__.tb_lineno if e.__traceback__ else None,
            traceback = traceback.format_tb(e.__traceback__) if e.__traceback__ else [],
        )

class GenResult(str,Enum):
    Success     = "Success"
    Timeout     = "Timeout"
    TooBig      = "TooBig"
    ImportError = "ImportError"
    Error       = "Error"
    RunError    = "RunError"
    UnknownErr  = "UnknownErr"
    NoLogErr    = "NoLogErr"

class ResultInfo(NamedTuple):
    time_taken: float
    memory_used: int
    stdout: str
    stderr: str

class Result(NamedTuple):
    # Filled in by the callee
    result_type: GenResult
    error: Union[ExceptionInfo,None]
    data: Union[ResultInfo,None]
    # Filled in by the caller
    module_path: Union[str,None] = None
    function_name: Union[str,None] = None
    output_file: Union[str,None] = None
    args: Union[argparse.Namespace,None] = None

    def _convert(self, item):
        """
        Recursively converts namedtuples to dictionaries, Enums to their values,
        and lists and dicts to their converted forms.
        """
        if isinstance(item, Enum):
            return item.value
        elif isinstance(item, tuple) and hasattr(item, '_asdict'):
            return {key: self._convert(value) for key, value in item._asdict().items()}
        elif isinstance(item, list):
            return [self._convert(sub_item) for sub_item in item]
        elif isinstance(item, dict):
            return {key: self._convert(value) for key, value in item.items()}
        else:
            return item

    def json(self):
        """
        Converts the namedtuple into JSON.
        """
        result_dict = self._convert(self)
        return json.dumps(
            result_dict,
            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
        )


# Context manager for timing out a function
class TimedExecution():
    def __init__(self, timeout):
        self.timeout = timeout
        self.timed_out = False
        self.start_time = None
        self.old_handler = None

    def __enter__(self):
        self.start_time = time.time()
        self.old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.timeout)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_handler)
        self.time_taken = time.time() - self.start_time
        self.timed_out = False
        return self

    def _handle_timeout(self, signum, frame):
        self.time_taken = time.time() - self.start_time
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
        self.mem_usage = None

    def __enter__(self):
        self.old_limit = resource.getrlimit(resource.RLIMIT_AS)
        # Only change the soft limit so that we can set it back
        resource.setrlimit(resource.RLIMIT_AS, (self.limit, self.old_limit[1]))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        resource.setrlimit(resource.RLIMIT_AS, self.old_limit)
        self.mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


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
        self.capture_stderr.__enter__()
        self.capture_stdout.__enter__()
        self.tempdir.__enter__()
        self.timeout.__enter__()
        self.memory_limit.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.memory_limit.__exit__(exc_type, exc_value, traceback)
        self.timeout.__exit__(exc_type, exc_value, traceback)
        self.tempdir.__exit__(exc_type, exc_value, traceback)
        self.capture_stdout.__exit__(exc_type, exc_value, traceback)
        self.capture_stderr.__exit__(exc_type, exc_value, traceback)

    def result(self) -> ResultInfo:
        return ResultInfo(
            time_taken = self.timeout.time_taken,
            memory_used = self.memory_limit.mem_usage,
            stdout = self.stdout.getvalue(),
            stderr = self.stderr.getvalue(),
        )

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
        output_file: str,
        function: Callable[[BinaryIO,BinaryIO],None],
        args: argparse.Namespace,
    ) -> Result:
    # Function takes a file-like BytesIO object (/dev/urandom)
    # and a writable BytesIO file object (output file)

    # Ensure the directory exists
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open('/dev/urandom', 'rb') as rng, \
        SizeLimitedBinaryFile(open(output_file, 'wb'),
                              max_size=args.size_limit) as out:
        try:
            with Sandbox(args.timeout, args.max_mem) as s:
                function(rng, out)
            return Result(
                result_type = GenResult.Success,
                error = None,
                data = s.result(),
            )
        except MemoryError as e:
            # Reset the memory limit immediately
            resource.setrlimit(resource.RLIMIT_AS, (-1,-1))
            return Result(
                result_type = GenResult.Error,
                error = ExceptionInfo.from_exception(e, args.module_path),
                data = s.result(),
            )
        except TooBigException:
            return Result(
                result_type = GenResult.TooBig,
                error = None,
                data = s.result(),
            )
        except TimeoutError as e:
            return Result(
                result_type = GenResult.Timeout,
                error = None,
                data = s.result(),
            )
        except Exception as e:
            return Result(
                result_type = GenResult.Error,
                error = ExceptionInfo.from_exception(e, args.module_path),
                data = s.result(),
            )

def get_function(module_path, function_name, args):
    try:
        # This needs to wrapped in the sandbox because modules can exec
        # code on load
        with Sandbox(args.timeout, args.max_mem) as s:
            full_module_path = os.path.abspath(module_path)
            shutil.copy(full_module_path, './generator_module.py')
            sys.path.append('.')
            import generator_module
            def capture_exit(rv=None):
                raise Exception(f"Attempted to exit with code {rv}")
            generator_module.exit = capture_exit
            generator_module.quit = capture_exit
            function = getattr(generator_module, function_name)
            return function
    except Exception as e:
        logger.info(f"Error importing module {module_path}: {e}")
        return Result(
            result_type = GenResult.ImportError,
            error = ExceptionInfo.from_exception(e, module_path),
            data = s.result(),
        )

def make_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('module_path', type=str,
        help='Path to the module containing the function to run')
    parser.add_argument(
        'function', type=str,
        help='The function to run',
    )
    parser.add_argument(
        '-n', '--num', type=int, default=1,
        help='Number of times to run the function')
    parser.add_argument(
        '-S', '--size-limit', type=int, default=50*1024*1024,
        help='Maximum size of the output file (in bytes)')
    parser.add_argument(
        '-o', '--output-prefix', type=str, default='./output',
        help='Output prefix')
    parser.add_argument(
        '-s', '--output-suffix', type=str, default='.dat',
        help='Output suffix')
    parser.add_argument(
        '-t', '--timeout', type=int, default=10,
        help='Timeout for the run (in seconds)')
    parser.add_argument(
        '-M', '--max-mem', type=int, default=1024*1024*1024,
        help='Maximum memory usage (in bytes)')
    parser.add_argument(
        '-L', '--logfile', type=str, default=None,
        help='Log file to write to')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def fill_result(result, module_path, function_name, output_file, args):
    return Result(
        result_type = result.result_type,
        error = result.error,
        data = result.data,
        module_path = module_path,
        function_name = function_name,
        output_file = output_file,
        args = args,
    )

def main():
    parser = make_parser('Run an input generator function in a loop')
    args = parser.parse_args()
    set_loglevel(logger, args)

    with open(args.logfile, 'w') if args.logfile else nullcontext(sys.stdout) as f:
        module_path = os.path.abspath(args.module_path)
        function_name = args.function
        function_or_result = get_function(module_path, function_name, args)
        if isinstance(function_or_result, Result):
            result = function_or_result = get_function(module_path, function_name, args)
            final_result = fill_result(result, module_path, function_name, None, args)
            print(final_result.json(), file=f)
            return

        function = function_or_result
        with ProcessPoolExecutor() as executor:
            futures = {}
            for i in range(args.num):
                output_file = f'{args.output_prefix}_{i:08}{args.output_suffix}'
                future = executor.submit(generate_one, output_file, function, args)
                futures[future] = output_file
            for future in as_completed(futures):
                output_file = futures[future]
                result = future.result()
                final_result = fill_result(result, module_path, function_name, output_file, args)
                print(final_result.json(), file=f)

if __name__ == '__main__':
    main()
