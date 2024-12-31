import os
import csv
import json
import argparse
import ast
import inspect


def save_args(args, filename='args.json'):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_args(filename='args.json'):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

def check_return_dict(func):
    """Decorator to enforce the return type of a function to be a dict."""
    def wrapper(self,*args, **kwargs):
       
        result = func(self,*args, **kwargs)
        # Check the return type
        if not isinstance(result, dict):
            raise TypeError(f"The method {func.__name__} must return a dictionary. Got {type(result).__name__} instead.")
        # Store the keys
        self.evaluation_keys = list(result.keys())
        # Return the result
        return result
    return wrapper


def get_dict_keys(func):
    """Extract keys of the dictionary returned by the function."""
    source = inspect.getsource(func)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            # Extract keys from the dictionary in the return statement
            keys = [
                key.s if isinstance(key, ast.Constant) else None  # For Python 3.8+
                for key in node.value.keys
            ]
            return keys
    return []


def write_to_csv(file, data):
    # Check if file exists to write the header only once
  
    file_exists = os.path.exists(file)
    
    with open(file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()  # Write headers only once
        writer.writerow({**data})

 