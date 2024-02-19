from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from functools import wraps
import time
from rich.progress import Progress, SpinnerColumn, TextColumn
from functools import wraps

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

# Example function that updates total dynamically and uses SpinnerColumn
@with_progress_bar
def simple_test_function(**kwargs):
    total_work = 10  # Example total work
    progress = kwargs.get('progress')
    task_id = kwargs.get('task_id')
    print(progress)

    if progress and task_id is not None:
        progress.update(task_id, total=total_work, description=f"[green]Executing task", advance=0)
        for i in range(total_work):
            time.sleep(1)  # Simulate work
            progress.update(task_id, advance=1)

if __name__ == "__main__":
    simple_test_function()  # Call the function