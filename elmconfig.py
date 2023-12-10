import argparse
from enum import Enum
import sys
from typing import Any, Dict, List
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Action, Namespace
from pathlib import Path, PosixPath
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from collections import OrderedDict

class SelectionStrategy(Enum):
    """Selection strategy"""
    Elites = 'elites'
    BestOfGeneration = 'best_of_generation'

class StoreDictKeyPair(Action):
    """Store a key-value pair in a dict"""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        d = namespace.__dict__.get(self.dest)
        if d is None:
            d = {}
        for kv in values:
            k, v = kv.split(':', 1)
            d[k] = v
        namespace.__dict__[self.dest] = d
    # Helper to turn a dict into a list of key-value pairs
    @staticmethod
    def invert(d: Dict) -> List[str]:
        return [ f"{k}:{v}" for k, v in d.items() ]

def flatten_conf(conf: Dict, prefix: str = '') -> Dict:
    # Flatten config dict, separating nested keys with '.'
    flat_conf = {}
    def _flatten(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v, prefix + k + '.')
            else:
                flat_conf[prefix + k] = v
    _flatten(conf, prefix)
    return flat_conf

def unflatten_conf(flat_conf: Dict) -> Dict:
    # Unflatten config dict, separating nested keys with '.'
    conf = {}
    for k, v in flat_conf.items():
        keys = k.split('.')
        d = conf
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return conf

def value_is_default(key: str, args: Namespace, parser: ArgumentParser) -> bool:
    """Check if a value is the default value for the corresponding option"""
    for opt in parser._get_optional_actions():
        if key == opt.dest:
            return args.__dict__[key] == opt.default
    return False

def convert_conf_item(key: str, val: Any, args: Namespace, parser: ArgumentParser):
    for opt in parser._get_optional_actions():
        if key == opt.dest:
            if opt.nargs is None:
                # Convert based on type
                conv = opt.type(val)
                # Set the value by calling the Action directly
                opt(parser, args, conv, f'--{key}')
            else:
                # Make sure val is a list
                if not isinstance(val, list):
                    print(f"Warning: {key} should be a list, but is {type(val)}; skipping", file=sys.stderr)
                    return
                # Convert each item in the list
                conv = [ opt.type(v) for v in val ]
                opt(parser, args, conv, f'--{key}')
            return
    # If we get here, we didn't find a matching option
    print(f"Warning: unknown option {key}; skipping", file=sys.stderr)

def merge_yaml_files(files: List[str]) -> CommentedMap:
    from yamlpath.merger import Merger, MergerConfig
    from yamlpath.commands.yaml_merge import get_doc_mergers, merge_docs
    from yamlpath.wrappers import ConsolePrinter
    from yamlpath.common import Parsers
    merge_args = Namespace(
        arrays='left', # Merge arrays by using the left value
        aoh='deep', # Merge arrays of hashes by merging hashes
        preserve_lhs_comments=True,
        debug=False, quiet=True, verbose=False
    )
    log = ConsolePrinter(merge_args)
    merge_config = MergerConfig(log, merge_args)
    yaml_editor: YAML = Parsers.get_yaml_editor()
    mergers: List[Merger] = []
    merge_count: int = 0
    # Process in reverse order because yamlpath only saves the comments
    # from the leftmost file
    for yaml_file in files[::-1]:
        if len(mergers) < 1:
            (mergers, mergers_loaded) = get_doc_mergers(
                log, yaml_editor, merge_config, yaml_file)
        else:
            exit_state = merge_docs(
                log, yaml_editor, merge_config, mergers, yaml_file)
            if not exit_state == 0:
                print(f"Error merging {yaml_file}", file=sys.stderr)
                break
        merge_count += 1
    dumps = []
    for doc in mergers:
        doc.prepare_for_dump(yaml_editor, '')
        dumps.append(doc.data)
    return dumps[0]

class ELMFuzzConfig:
    """ELMFuzz configuration"""
    def __init__(self, default_config_file: str = 'config.yaml'):
        self.init_parser()
        self.yaml = YAML(typ='rt')
        self.yaml.preserve_quotes = True
        self.default_config_file = default_config_file
        # Defer loading config file until parse_args() is called, because
        # a config file may be specified on the command line
        self.config = None
        self.init_dumper()

    def init_dumper(self) -> None:
        self.yaml.representer.add_representer(
            PosixPath,
            lambda r, v: r.represent_str(str(v))
        )
        self.yaml.default_flow_style = False

    def init_parser(self) -> ArgumentParser:
        self.parser = ArgumentParser()
        self.parser_groups = OrderedDict()
        group = self.parser.add_argument_group('Global options')
        group.add_argument("--config", default=None, type=Path,
                           help="Path to config file (overrides default search)")
        group.add_argument("--afl_dir", type=Path, help="Path to AFL++ directory (for afl-showmap)")
        group.add_argument("--target.srcs", type=Path, nargs='+', action='extend',
                           help="Source files in the target")
        group.add_argument("--target.covbin", type=Path,
                           help="Path to the target binary with coverage instrumentation")
        group.add_argument("--model.names", type=str, nargs='+', action='extend', help="List of model names")
        group.add_argument("--model.endpoints", type=str, nargs='+', action=StoreDictKeyPair,
                           metavar="NAME:ENDPOINT", help="List of model endpoints, formatted as name:endpoint")
        group.add_argument("--run.seeds", type=Path, nargs='+', action='extend',
                           help="Seed files (generator programs that will be mutated)")
        group.add_argument("--run.num_generations", type=int, default=10, help="Number of generations to run")
        group.add_argument("--run.selection_strategy", type=str, default='elites', help="Selection strategy",
                           choices=[s.value for s in SelectionStrategy])
        group.add_argument("--run.num_selected", type=int, default=10,
                           help="Number of seeds to select each generation")
        self.parser_groups['global'] = group

    def parse_args(self) -> Namespace:
        self.args = self.parser.parse_args()
        self.config = self.load_config(self.args)
        self.add_config_args(self.args)
        return self.args

    def load_config(self, args: Namespace) -> None:
        # Load config file(s)
        if args.config is not None:
            conf = self.yaml.load(args.config)
        else:
            to_merge = [ config_file for config_file in self.config_file_search() if os.path.exists(config_file) ]
            if len(to_merge) == 1:
                conf = self.yaml.load(open(to_merge[0]).read())
            elif len(to_merge) > 1:
                conf = merge_yaml_files(to_merge)
            else:
                conf = CommentedMap({})
        self.config = conf
        return conf

    def config_file_search(self) -> List[str]:
        # NB: The order matters here because later files override earlier files
        config_files = []
        # Check script dir
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_files.append(
            os.path.join(script_dir, self.default_config_file)
        )
        # Check CWD
        config_files.append(self.default_config_file)
        # Check ELMFUZZ_RUNDIR env var
        if 'ELMFUZZ_RUNDIR' in os.environ:
            config_files.append(
                os.path.join(os.environ['ELMFUZZ_RUNDIR'], self.default_config_file)
            )
        # Check ELMFUZZ_CONFIG env var
        if 'ELMFUZZ_CONFIG' in os.environ:
            config_files.append(
                os.environ['ELMFUZZ_CONFIG']
            )
        return config_files

    def add_config_args(self, args: Namespace) -> None:
        """Add config arguments to an existing parser"""
        # Flatten config dict
        self.config_flat = flatten_conf(self.config)
        for k, v in self.config_flat.items():
            if k in args.__dict__:
                if value_is_default(k, args, self.parser):
                    convert_conf_item(k, v, args, self.parser)
                else:
                    print(f"Note: ignoring config value for {k} because it was set on the command line", file=sys.stderr)
            else:
                print(f"Ignored unknown parameter {k} in yaml.", file=sys.stderr)
        # Set default values for any options that were not set
        for opt in self.parser._get_optional_actions():
            if opt.dest not in args.__dict__:
                args.__dict__[opt.dest] = opt.default

    def unflatten_conftuple(self, conf: List) -> CommentedMap:
        nested_dict = CommentedMap()

        for compound_key, value, htxt in conf:
            keys = compound_key.split('.')
            current_level = nested_dict

            for key in keys[:-1]:
                # Create a new dictionary if the key doesn't exist
                if key not in current_level:
                    current_level[key] = CommentedMap()
                # Move to the next level
                current_level = current_level[key]

            # Set the value at the final level
            current_level[keys[-1]] = value
            current_level.yaml_set_comment_before_after_key(keys[-1], before=htxt)

        return nested_dict

    def dump_config(self,
            args: Namespace,
            add_comments=True,
            skip_defaults=True,
        ) -> str:
        """Dump config to a YAML file"""
        # For getting the help text
        fmt = ArgumentDefaultsHelpFormatter('ELMFuzz', indent_increment=0, width=sys.maxsize)
        def get_help_text(opt):
            return fmt._get_help_string(opt) % opt.__dict__ if add_comments else None
        # Get all options with their values and help text
        conf = []
        for opt in self.parser._get_optional_actions():
            # Skip options that are not set
            if opt.dest not in args.__dict__:
                continue
            # Skip suppressed options
            if opt.default == argparse.SUPPRESS:
                continue
            # Skip options that are set to the default value, if requested
            if not skip_defaults and value_is_default(opt.dest, args, self.parser):
                continue
            # Get the value
            val = args.__dict__[opt.dest]
            # If the action has an inverse, use that to convert the value
            if hasattr(opt, 'invert'):
                val = opt.invert(val)
            # Add to config dict
            conf.append((opt.dest, val, get_help_text(opt)))
        # Unflatten config dict
        conf_dict = self.unflatten_conftuple(conf)
        # Dump to YAML
        self.yaml.dump(conf_dict, sys.stdout)


if __name__ == "__main__":
    elmconf = ELMFuzzConfig()
    args = elmconf.parse_args()
    elmconf.dump_config(args)
    # parser = default_parser()
    # args = parser.parse_args()
    # add_config_args(parser, args)
    # for k, v in args.__dict__.items():
    #     print(f"{k}: {v} ({type(v).__name__})")
    # yaml = YAML(typ='rt')
    # doc = merge_yaml_files(sys.argv[1:])
    # yaml.dump(doc, sys.stdout)
