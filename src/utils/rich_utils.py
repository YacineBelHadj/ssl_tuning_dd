from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from functools import wraps

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "backbone",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
            components are printed.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra
            output folder.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)




def with_progress_bar(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__  # Extract the function's name
        # Create a progress bar with the function's name in the description
        with Progress(SpinnerColumn(), 
                      TextColumn(f"[bold green]Executing: [not bold]{func_name}"), 
                      TaskProgressColumn(),
                      BarColumn(),
                      transient=True) as progress:
            task = progress.add_task(f"Starting {func_name}...", total=100)
        
            kwargs['progress'] = progress
            kwargs['task_id'] = task
            
            result = func(*args, **kwargs)
            
            # Update progress to complete, ensuring it shows as finished
            progress.update(task, completed=100)
            progress.remove_task(task)
            
        return result
    return wrapper
