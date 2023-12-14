#!/usr/bin/env python3

import argparse
import copy
from datetime import datetime
from enum import Enum
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Action, Namespace
from pathlib import Path, PosixPath
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from collections import OrderedDict
from collections.abc import Sequence as SequenceABC
from io import StringIO

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
        if d is None:
            return None
        return [ f"{k}:{v}" for k, v in d.items() ]

def value_is_default(key: str, args: Namespace, parser: ArgumentParser) -> bool:
    """Check if a value is the default value for the corresponding option"""
    for opt in parser._get_optional_actions():
        if key == opt.dest:
            return args.__dict__[key] == opt.default
    return False

def convert_conf_item(key: str, val: Any, args: Namespace, parser: ArgumentParser):
    store_actions = {
        act for name, act in parser._registries['action'].items()
        if isinstance(name, str) and name.startswith('store_')
    }
    for opt in parser._get_optional_actions():
        def type_conv(v):
            if opt.type is None:
                return v
            elif v is None:
                return None
            else:
                return opt.type(v)
        if key == opt.dest:
            # If the action is one that stores a value, just set the value
            if type(opt) in store_actions:
                args.__dict__[key] = val
            # If the action is one that returns a list, make sure val is a list
            elif opt.nargs in ['+', '*'] or isinstance(opt.nargs, int) and opt.nargs > 1:
                # Make sure val is a list
                if not isinstance(val, list):
                    print(f"Warning: {key} should be a list, but is {type(val)}; skipping", file=sys.stderr)
                    return
                # Convert each item in the list
                conv = [ type_conv(v) for v in val ]
                opt(parser, args, conv, f'--{key}')
            else:
                conv = type_conv(val)
                # Set the value by calling the Action directly
                opt(parser, args, conv, f'--{key}')
            return
    # If we get here, we didn't find a matching option
    print(f"Warning: unknown option {key}; skipping", file=sys.stderr)

def nest_namespace(ns: Namespace) -> Namespace:
    """Recursively create a nested namespace from a flat one"""
    # Make a copy of the namespace
    nested = Namespace(**ns.__dict__)
    for k, v in ns.__dict__.items():
        if '.' in k:
            # Split the key
            nested_name, rest = k.split('.', 1)
            # Create the nested namespace if it doesn't exist
            if not hasattr(nested, nested_name):
                setattr(nested, nested_name, Namespace())
            # Set the value
            setattr(getattr(nested, nested_name), rest, v)
            # Delete the old value
            delattr(nested, k)
    # Recurse
    for k, v in nested.__dict__.items():
        if isinstance(v, Namespace):
            setattr(nested, k, nest_namespace(v))
    return nested
class ELMFuzzConfig:
    """ELMFuzz configuration"""
    def __init__(self,
                 prog: str = None,
                 default_config_file: str = 'config.yaml',
                 parents: Dict[str,ArgumentParser] = None
                 ):
        if prog is None:
            prog = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.prog = prog
        self.default_config_file = default_config_file
        self.parent_parsers = parents if parents is not None else {}
        # Used so that we can add help text to subgroups, e.g.
        # the "target" group which has target.covbin, target.srcs, etc.
        self.subgroup_help = {}
        self.init_parser()
        # Defer loading config file until parse_args() is called, because
        # a config file may be specified on the command line
        self.config = None
        self.init_dumper()

    def init_dumper(self) -> None:
        self.yaml = YAML(typ='rt')
        self.yaml.preserve_quotes = True
        self.yaml.representer.add_representer(
            PosixPath,
            lambda r, v: r.represent_str(str(v))
        )

    def merge_yaml_files(self, files: List[str]) -> CommentedMap:
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

    def parse_args_nofail(self, args=None, namespace=None) -> Namespace:
        # Helper to remove an option from the parser
        def _remove_argument(parser, arg):
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
        # Work on a copy of the parser, args, and namespace
        dump_parser = copy.deepcopy(self.parser)
        if args is None:
            dump_args = sys.argv[1:][:]
        else:
            dump_args = copy.deepcopy(args)
        dump_namespace = copy.deepcopy(namespace)
        _remove_argument(dump_parser, '--dump-config')

        # Remove the option from the args; tricky since nargs is '?'. We'll do it
        # the dumb way and just see if arg+1 contains any of the strings we expect
        dump_config_index = None
        for i, arg in enumerate(dump_args):
            if arg.startswith('--dump-config'):
                dump_config_index = i
                break
        assert dump_config_index is not None, "Couldn't find --dump-config in args"
        dump_option = dump_args[dump_config_index]
        if '=' in dump_option or len(dump_args) == dump_config_index + 1:
            # Remove the whole thing
            del dump_args[dump_config_index]
        else:
            next_arg = dump_args[dump_config_index + 1]
            expected_params = { 'skip_comments', 'skip_defaults', 'file' }
            if any(e in next_arg for e in expected_params):
                # Remove both
                del dump_args[dump_config_index:dump_config_index+2]
            else:
                # Remove just the option
                del dump_args[dump_config_index]

        # Make all arguments optional
        for action in dump_parser._actions:
            action.required = False

        # Parse again
        args = dump_parser.parse_args(dump_args, dump_namespace)
        # Load config file
        conf = self.load_config(args)
        # Add config args
        self.add_config_args(args)
        return args

    def dump_config_action(self,
                parser: ArgumentParser,
                namespace: Namespace,
                values: Any,
                option_string: str,
            ) -> None:
        # The action is invoked during parsing, so the args aren't fully set up yet.
        # Unfortunately, if we do a full parse, then any missing required arguments
        # will raise errors. So we make a copy of the parser, change its required
        # arguments to optional, and parse the args again. Also, we need to remove the
        # --dump-config option from the parser and the arguments, or we'll either
        # recurse or get an error about an unknown option. Oof!
        args = self.parse_args_nofail(
            self._most_recent_args,
            self._most_recent_namespace,
        )

        # Dump config
        kwargs = {}
        if values is True:
            # No arguments
            pass
        else:
            # Split comma-separated arguments
            dump_opts = values.split(',')
            for opt in dump_opts:
                if opt.startswith('file='):
                    kwargs['file'] = opt.split('=')[1]
                elif opt in ['skip_comments', 'skip_defaults']:
                    kwargs[opt] = True
                else:
                    self.parser.error(f"Unknown argument to --dump-config: {opt}")
        # Dump to file if requested
        if 'file' in kwargs:
            kwargs['file'] = open(kwargs['file'], 'w')
        else:
            kwargs['file'] = sys.stdout
        # Dump
        self.dump_config(args, **kwargs)
        sys.exit(0)

    def init_parser(self) -> ArgumentParser:
        # First try without a conflict handler so we can
        # print a warning if there are any conflicts
        self.parser = ArgumentParser(
            parents=self.parent_parsers.values(),
            prog=self.prog,
            add_help=not bool(self.parent_parsers),
        )
        config_group = self.parser.add_argument_group('Config options')
        config_group.add_argument("--config", default=None, type=Path,
                                  help="Path to config file (overrides default search)")

        def make_dump_config_action(parent):
            class DumpConfigAction(Action):
                def __init__(self, option_strings, dest, nargs=None, **kwargs):
                    super().__init__(option_strings, dest, nargs=nargs, **kwargs)
                    self.parent = parent
                def __call__(self, parser, namespace, values, option_string=None):
                    self.parent.dump_config_action(parser, namespace, values, option_string)
            return DumpConfigAction
        config_group.add_argument(
            '--dump-config', type=str, nargs='?',
            help=("Dump config and exit.  "
                  "The optional (comma-separated) argument controls how the config is dumped: "
                  "skip_comments (Don't add help text as comments), "
                  "skip_defaults (Skip options that are set to their default value), and"
                  "file=<path> (Dump to the specified file instead of stdout)."
            ),
            default=False,
            const=True,
            action=make_dump_config_action(self),
        )

        group = self.parser.add_argument_group('Global options')
        self.subgroup_help['target'] = 'Options to configure the target program being fuzzed'
        group.add_argument("--target.srcs", type=Path, nargs='+', action='extend',
                           help="Source files in the target")
        group.add_argument("--target.covbin", type=Path,
                           help="Path to the target binary with coverage instrumentation")
        self.subgroup_help['model'] = 'Options to configure the model(s) used for variant generation'
        group.add_argument("--model.names", type=str, nargs='+', action='extend', help="List of model names")
        group.add_argument("--model.endpoints", type=str, nargs='+', action=StoreDictKeyPair,
                           metavar="NAME:ENDPOINT", help="List of model endpoints, formatted as name:endpoint")
        self.subgroup_help['run'] = 'Options to configure the run of the evolutionary algorithm'
        group.add_argument("--run.seeds", type=Path, nargs='+', action='extend',
                           help="Seed files (generator programs that will be mutated)")
        group.add_argument("--run.num_generations", type=int, default=10, help="Number of generations to run")
        selection_choices = [ s.value for s in SelectionStrategy ]
        selection_choices_str = ', '.join(selection_choices)
        group.add_argument("--run.selection_strategy", type=str, default='elites',
                           help=f"Selection strategy (one of: {selection_choices_str})",
                           choices=selection_choices)
        group.add_argument("--run.num_selected", type=int, default=10,
                           help="Number of seeds to select each generation")
        group.add_argument("--run.genvariant_dir", type=str,
                           default='{ELMFUZZ_RUNDIR}/{GEN}/variants/{MODEL}',
                           help="Directory (template) to store generated variants")
        group.add_argument("--run.genoutput_dir", type=str,
                            default='{ELMFUZZ_RUNDIR}/{GEN}/outputs/{MODEL}',
                            help="Directory (template) to store generated outputs")
        group.add_argument("--run.logdir", type=str,
                            default='{ELMFUZZ_RUNDIR}/{GEN}/logs',
                            help="Directory (template) to store logs")
        group.add_argument("--run.clean", action='store_true',
                            help="Clean the output directories before running")

        # XXX: For testing only
        # group = self.parser.add_argument_group('Test options', 'Lorem ipsum dolor sit amet')
        # self.subgroup_help['test_a'] = 'Test options for group A'
        # group.add_argument("--test_a.boolopt", action='store_true', help="Test option A (bool)")
        # group.add_argument("--test_a.stropt", type=str, help="Test option A (str)")
        # self.subgroup_help['test_b'] = 'Test options for group B'
        # group.add_argument("--test_b.intopt", type=int, help="Test option B (int)")
        # group.add_argument("--test_b.floatopt", type=float, help="Test option B (float)")

    def parse_args(self,
                   args : Optional[Sequence[str]] = None,
                   namespace : Optional[Namespace] = None,
                   nested : bool = True,
                   ) -> Namespace:
        """Parse arguments

        :param args: Arguments to parse (default: sys.argv[1:])
        :param namespace: Namespace to store the parsed arguments (default: create a new one)
        :param nested: Whether the returned namespace should be nested to convert key names
            like foo.bar into foo = Namespace(bar=...) (default: True)

        Returns a namespace with the parsed arguments.
        """
        # HACK: save the most recently parsed argv/namespace so that dump_config_action
        # can access them
        self._most_recent_args = args
        self._most_recent_namespace = namespace
        self.args = self.parser.parse_args(args, namespace)
        self.config = self.load_config(self.args)
        self.add_config_args(self.args)

        # For convenience, nest the namespace we return
        # We keep the original namespace in self.args
        if nested:
            return nest_namespace(self.args)
        else:
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
                conf = self.merge_yaml_files(to_merge)
            else:
                conf = CommentedMap()
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

    @staticmethod
    def flattened_conf(conf: Dict, prefix='', flatten_lists=False) -> Dict:
        # Flatten config dict, separating nested keys with '.'
        flat_conf = {}
        def _flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    _flatten(v, prefix + k + '.')
                elif isinstance(v, list):
                    if flatten_lists:
                        for i, item in enumerate(v):
                            _flatten({str(i): item}, prefix + k + '.')
                    else:
                        flat_conf[prefix + k] = v
                else:
                    flat_conf[prefix + k] = v
        _flatten(conf, prefix)
        return flat_conf

    def add_config_args(self, args: Namespace) -> None:
        """Add config arguments to an existing parser"""
        if self.config is None:
            raise RuntimeError("Config not loaded; call parse_args() first")
        # Flatten config dict
        config_flat = self.flattened_conf(self.config)
        for k, v in config_flat.items():
            # CLI options are in their own section, so will appear as
            #   cli.<prog>.<option>
            # But the current parser will only have <option>, so we need to
            # strip the prefix
            if k.startswith('cli.'):
                if k.startswith(f'cli.{self.prog}.'):
                    # Strip the prefix
                    norm_k = k[len(f'cli.{self.prog}.'):]
                else:
                    # Skip options for other utilities
                    continue
            else:
                norm_k = k

            if norm_k in args.__dict__:
                if value_is_default(norm_k, args, self.parser):
                    convert_conf_item(norm_k, v, args, self.parser)
                else:
                    pass
                    # print(f"Note: ignoring config value for {k} because it was set on the command line", file=sys.stderr)
            else:
                print(f"Ignored unknown parameter {k} in yaml.", file=sys.stderr)

    def unflatten_conf(
            self,
            conf: Dict[str,List[Tuple[str,Any,str]]],
            header_comment: bool = True,
            base_indent: int = 0,
            **kwargs
            ) -> CommentedMap:
        """Unflatten a config dict by section

        :param conf: A dict of config options, grouped by section. The values are
            a list of tuples (key, value, help_text).
        :param kwargs: Arguments that were passed to dump_config() (used only for the "generated by" comment)
        """
        existing_configs = [ config_file for config_file in self.config_file_search() if os.path.exists(config_file) ]
        now = datetime.now().strftime('%F %r')
        nested_dict = CommentedMap()
        if header_comment:
            config_comment = (
                f"Automatically generated by {self.prog}.dump_config() on {now} with options:\n"
                f"    {kwargs}\n"
                f"Based on existing config file(s): " + ', '.join(existing_configs)
            )
            nested_dict.yaml_set_start_comment(config_comment, indent=0)

        # Helper to get the section comment
        def _section_comment(section_name: str) -> str:
            if kwargs['skip_comments']:
                return None
            if section_name == 'options':
                return None
            if section_name in self.parent_parsers:
                return None
            # Get the group description if it exists
            desc = None
            for group in self.parser._action_groups:
                if group.title == section_name:
                    desc = group.description
                    break
            if desc is None:
                section_str = section_name
            else:
                section_str = f"{section_name}: {desc}"
            section_str = ' ' + section_str + ' '
            return section_str.center(80-base_indent, '-')

        # Amount of indentation for map keys
        map_indent = self.yaml.map_indent if self.yaml.map_indent is not None else 2

        for section_name, section_conf in conf.items():
            if section_name == 'cli':
                continue
            first = True
            for compound_key, value, htxt in section_conf:
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
                if first:
                    # Set the section comment at the first key
                    section_header = _section_comment(section_name)
                    if section_header is not None:
                        section_header = '\n' + section_header
                        nested_dict.yaml_set_comment_before_after_key(keys[0], before=section_header)
                    first = False
                # Set the help text as a comment
                if htxt:
                    indent = base_indent + map_indent * (len(keys) - 1)
                    help_text = '\n'.join(textwrap.wrap(htxt, width=80-indent))
                    current_level.yaml_set_comment_before_after_key(keys[-1], before=help_text, indent=indent)

        if 'cli' in conf:
            # Add CLI options
            cli_section = CommentedMap()
            for cli_name, cli_conf in conf['cli'].items():
                # Recurse
                cli_unflattened = self.unflatten_conf(
                    {cli_name: cli_conf},
                    header_comment=False,
                    base_indent=map_indent*2,
                    **kwargs
                )
                # Skip empty CLI sections
                if not cli_unflattened:
                    continue
                cli_section[cli_name] = cli_unflattened
                if not kwargs.get('skip_comments', False):
                    # Add a comment before the utility's args with the description
                    cli_comment = f'{cli_name}: {self.parent_parsers[cli_name].description}'
                    cli_section.yaml_set_comment_before_after_key(
                        cli_name,
                        before=cli_comment,
                        indent=map_indent,
                    )
            if cli_section:
                nested_dict['cli'] = cli_section
                if not kwargs.get('skip_comments', False):
                    cli_header = ' ' + 'Specific CLI utility options' + ' '
                    cli_header = cli_header.center(80, '-')
                    cli_header = '\n' + cli_header + '\n\n'
                    nested_dict.yaml_set_comment_before_after_key(
                        'cli',
                        before=cli_header
                    )

        # Finally, add the per-section help text
        if not kwargs.get('skip_comments', False):
            for subgroup, sg_help in self.subgroup_help.items():
                keys = subgroup.split('.')
                current_level = nested_dict
                for key in keys[:-1]:
                    current_level = current_level[key]
                indent = base_indent + map_indent * (len(keys) - 1)
                sg_help = '\n'.join(textwrap.wrap(sg_help, width=80-indent))
                sg_help = '\n' + sg_help
                current_level.yaml_set_comment_before_after_key(keys[-1], before=sg_help, indent=indent)
        return nested_dict

    def get_config(self,
            args: Namespace,
            skip_comments=False,
            skip_defaults=False,
            skip_names=['config', 'dump_config'],
        ) -> CommentedMap:
        # For getting the help text
        fmt = ArgumentDefaultsHelpFormatter('ELMFuzz', indent_increment=0, width=sys.maxsize)
        def get_help_text(option):
            if skip_comments:
                return None
            if option.help is None:
                return None
            help_text = fmt._get_help_string(option) % option.__dict__
            return help_text

        # What options to skip
        def skip_option(option):
            # Skip options that are not set
            if option.dest not in args.__dict__:
                # print(f'Skipping {option.dest} because it is not set', file=sys.stderr)
                return True
            # Skip options that are in the skip list
            if option.dest in skip_names:
                # print(f'Skipping {option.dest} because it is in the skip list', file=sys.stderr)
                return True
            # Skip suppressed options
            if option.default == argparse.SUPPRESS or option.help == argparse.SUPPRESS:
                # print(f'Skipping {option.dest} because it is suppressed', file=sys.stderr)
                return True
            # Skip positional arguments
            if not option.option_strings:
                # print(f'Skipping {option.dest} because it is a positional argument', file=sys.stderr)
                return True
            # Skip options that are set to the default value, if requested
            if skip_defaults and value_is_default(option.dest, args, self.parser):
                return True
            return False

        # Get a set of dest names specific to a CLI utility (i.e., not the global options)
        utility_opt_names = {
            opt.dest
            for parent_parser in self.parent_parsers.values()
            for opt in parent_parser._get_optional_actions()
        }

        # Get all options with their values and help text
        conf_sections = OrderedDict()
        for group in self.parser._action_groups:
            if group.title == 'positional arguments':
                continue
            conf = []
            for opt in group._group_actions:
                if skip_option(opt):
                    continue
                # Skip utility options (we will add them later in the CLI section)
                if opt.dest in utility_opt_names:
                    continue
                # Get the value
                val = args.__dict__[opt.dest]
                # If the action has an inverse, use that to convert the value
                if hasattr(opt, 'invert'):
                    val = opt.invert(val)
                # Add to config dict
                conf.append((opt.dest, val, get_help_text(opt)))
            conf_sections[group.title] = conf

        # Add CLI options
        conf_sections['cli'] = {}
        for parent_name, parent_parser in self.parent_parsers.items():
            conf = []
            for opt in parent_parser._get_optional_actions():
                if skip_option(opt):
                    continue
                # Get the value
                val = args.__dict__[opt.dest]
                # If the action has an inverse, use that to convert the value
                if hasattr(opt, 'invert'):
                    val = opt.invert(val)
                # Add to config dict
                conf.append((opt.dest, val, get_help_text(opt)))
            conf_sections['cli'][parent_name] = conf

        # Unflatten config dict
        conf_dict = self.unflatten_conf(
            conf_sections,
            skip_comments=skip_comments,
            skip_defaults=skip_defaults,
            skip_names=skip_names,
        )
        return conf_dict

    def dump_config(self,
            args: Namespace,
            skip_comments=False,
            skip_defaults=False,
            skip_names=['config', 'dump_config'],
            file : Optional[TextIO] = None,
        ) -> Optional[str]:
        """Dump config to a YAML file

        :param args: The parsed arguments
        :param skip_comments: Whether to skip adding help text as comments
        :param skip_defaults: Whether to skip options that are set to their default value
        :param skip_names: Names of options to skip
        :param file: The file to dump to (default: return a string)
        """
        # Get config dict
        conf_dict = self.get_config(
            args,
            skip_comments=skip_comments,
            skip_defaults=skip_defaults,
            skip_names=skip_names,
        )
        # Dump to YAML
        if file is not None:
            self.yaml.dump(conf_dict, file)
            return None
        else:
            s = StringIO()
            self.yaml.dump(conf_dict, s)
            return s.getvalue()

    def dumps(self, **kwargs) -> str:
        """Dump config to a YAML string

        :param kwargs: Arguments to pass to dump_config()
        """
        return self.dump_config(self.args, file=None, **kwargs)

    def dump(self, file=None, **kwargs) -> None:
        """Dump config to a YAML file
        :param file: The file to dump to (default: sys.stdout)
        :param kwargs: Arguments to pass to dump_config()
        """
        if file is None:
            file = sys.stdout
        self.dump_config(self.args, file=file, **kwargs)

    def __repr__(self) -> str:
        if self.config is not None:
            return f"ELMFuzzConfig(prog={self.parser.prog}, {self.args})"
        else:
            return f"ELMFuzzConfig(prog={self.parser.prog}, [not yet parsed])"

def get_config_for_progs(progs: List[str], **kwargs) -> CommentedMap:
    full_config = None
    for prog in progs:
        module = __import__(prog)
        parser = module.make_parser()
        config = ELMFuzzConfig(
            prog=prog,
            parents={prog: parser}
        )
        module.init_parser(config)
        prog_args = config.parse_args_nofail(['--dump-config'])
        conf_dict = config.get_config(prog_args, **kwargs)
        if full_config is None:
            full_config = conf_dict
        else:
            full_config['cli'][prog] = conf_dict['cli'][prog]
    return full_config

ALL_PROGS = ['genvariants_parallel', 'genoutputs', 'getcov']
def defaultconfig_cmd(args):
    options = {
        k : v for k, v in args.__dict__.items()
        if k.startswith('skip_')
    }
    full_config = get_config_for_progs(args.progs, **options)
    if not full_config:
        print("Error: no config options found", file=sys.stderr)
        sys.exit(1)
    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.representer.add_representer(
        PosixPath,
        lambda r, v: r.represent_str(str(v))
    )
    yaml.dump(full_config, args.file)

class Raise: pass
class Parent: pass
def mget(d, keys, default=None):
    for i, key in enumerate(keys):
        # Handle list indices
        if key.isdigit():
            key = int(key)
        try:
            d = d[key]
        except (KeyError, IndexError) as e:
            if default is Raise:
                raise e
            elif default is Parent:
                return keys[:i], d
            else:
                return default
    if default is Parent:
        return keys, d
    else:
        return d

def get_cmd(args):
    def expand_path(val):
        return str(val.expanduser() if not args.no_expand else val)
    def conv(val):
        val_converters = [
            (Path, expand_path),
            (str, lambda v: v),
            (SequenceABC, lambda v: ' '.join([conv(x) for x in v])),
        ]
        for ty, conv_func in val_converters:
            if isinstance(val, ty):
                return conv_func(val)
        return val
    conf_dict = get_config_for_progs(args.progs)
    keys = args.key.split('.')
    try:
        val = mget(conf_dict, keys, Raise)
    except (KeyError, IndexError):
        print(f"Error: {args.key} is not a valid key", file=sys.stderr)
        sys.exit(1)
    val = conv(val)
    if isinstance(val, str) and not args.no_subst:
        # Do any substitutions
        subst_dict = dict([ s.split('=', 1) for s in args.substitutions ])
        # Expand environment variables
        if not args.no_env:
            subst_dict.update(os.environ)
        val = val.format(**subst_dict)
    print(val)

def list_cmd(args):
    conf_dict = get_config_for_progs(args.progs)
    keys = args.prefix.split('.')
    matching, sub_dict = mget(conf_dict, keys, Parent)
    if not isinstance(sub_dict, dict) and not isinstance(sub_dict, list):
        # Must be a value
        print('.'.join(matching))
        return
    # We'll use left_over to do string matching
    left_over = keys[len(matching):]
    if len(left_over) > 1:
        print(f"Error: {args.prefix} is not a valid prefix", file=sys.stderr)
        sys.exit(1)
    left_over = left_over[0] if left_over else ''
    # Flatten the dict
    if matching:
        flat_prefix = '.'.join(matching) + '.'
    else:
        flat_prefix = ''
    flat_dict = ELMFuzzConfig.flattened_conf(sub_dict, prefix=flat_prefix, flatten_lists=True)
    # Filter by prefix
    matching = [ k for k in flat_dict.keys() if k.startswith(left_over) ]
    if not matching:
        print(f"{args.prefix} does not match any keys", file=sys.stderr)
        # Not an error
        sys.exit(0)
    # Print matching keys
    for k in matching:
        print(k)

def main():
    parser = argparse.ArgumentParser(description="ELMFuzz configuration utility")
    parser.add_argument('--config', type=Path, help="Path to config file (overrides default search)")
    parser.add_argument('-p', '--prog', type=str, action='append',
                        dest='progs',
                        help="Select programs",
                        choices=ALL_PROGS, default=None)
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    cmd = subparsers.add_parser('dumpconfig', help="Dump config to YAML")
    cmd.add_argument('--skip-comments', action='store_true',
                     default=argparse.SUPPRESS,
                     help="Skip adding help text as comments")
    cmd.add_argument('--skip-defaults', action='store_true',
                     default=argparse.SUPPRESS,
                     help="Skip options that are set to their default value")
    cmd.add_argument('--skip-names', type=str, nargs='+',
                     default=argparse.SUPPRESS,
                     help="Names of options to skip")
    cmd.add_argument('--file', type=argparse.FileType('w'), default=sys.stdout,
                     help="File to dump to (default: stdout)")
    cmd.set_defaults(func=defaultconfig_cmd)
    cmd = subparsers.add_parser('get', help="Get the value of a config option")
    cmd.add_argument('key', type=str, help="Config option to get, e.g. model.endpoints.0")
    cmd.add_argument('--no-subst', action='store_true', help="Don't do any substitutions")
    cmd.add_argument('--no-expand', action='store_true', help="Don't expand ~ in paths")
    cmd.add_argument('--no-env', action='store_true', help="Don't expand environment variables")
    cmd.add_argument('-s', '--substitute', type=str, action='append', dest='substitutions',
                     default=[], metavar="VAR=VAL", help="Substitute VAR with VAL in strings")
    cmd.set_defaults(func=get_cmd)
    cmd = subparsers.add_parser('list', help="List config options")
    cmd.add_argument('prefix', type=str, nargs='?', default='', help="Prefix to filter options")
    cmd.set_defaults(func=list_cmd)
    args = parser.parse_args()
    if args.progs is None:
        args.progs = ALL_PROGS
    args.func(args)

if __name__ == "__main__":
    main()
