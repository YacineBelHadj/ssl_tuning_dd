from functools import wraps
from typing import Any, Callable
import os
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import argparse

def get_args_parser() -> argparse.ArgumentParser:
    """Get parser for additional Hydra's command line flags."""
    parser = argparse.ArgumentParser(
        description="Additional Hydra's command line flags parser."
    )

    parser.add_argument(
        "--config-path",
        "-cp",
        nargs="?",
        default=None,
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        nargs="?",
        default=None,
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        nargs="?",
        default=None,
        help="Adds an additional config dir to the config search path",
    )
    return parser

def set_env(val,name) -> None:
    # check type of val
    val = hydra.utils.instantiate(val)
    os.environ[name] = str(val)
    print(os.environ[name])


def resolve_env(version_base: str,
        config_path: str,
        config_name: str) -> Callable:
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    if args.config_path is not None:
        config_path = args.config_path
    if args.config_dir is not None:
        config_path = args.config_dir
    if args.config_name is not None:
        config_name = args.config_name 
    if not OmegaConf.has_resolver('set_env'):
        with initialize_config_dir(
            version_base=version_base,config_dir=config_path):
            cfg = compose(config_name=config_name)
        OmegaConf.register_new_resolver("set_env", set_env)
        OmegaConf.resolve(cfg)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator
