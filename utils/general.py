
import numpy as np
import os
import pickle
from typing import List

import pandas as pd


def setup_seed(seed=9899):
    import random
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_uri_writer_processor(data, path: str, **kwargs):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if path.endswith('pkl') or path.endswith('pickle'):
        pickle.dump(data, open(path, 'wb'))
    else:
        # treat file as csv file
        data.to_csv(path, index=False)
    return path


def file_uri_reader_processor(uri, columns=None, **kwargs):
    if uri.endswith('csv'):
        data = pd.read_csv(uri, sep='\n', delimiter=',', usecols=columns)
    elif uri.endswith('pkl') or uri.endswith('pickle'):
        with open(uri, 'rb') as file:
            try:
                data = pickle.load(file, encoding='latin-1')
            except:
                data = pickle.load(file)
    else:
        # treat file as csv file
        data = pd.read_csv(uri, sep='\n', delimiter=',', usecols=columns)
    return data


def list_of_dict_to_dict(list_of_dicts):
    if not list_of_dicts:
        raise ValueError("The list of dicts is empty")

    dict_of_lists = {key: np.array([d[key] for d in list_of_dicts]) for key in list_of_dicts[0]}

    return dict_of_lists


def get_value_by_key(key, list_dict):
    return [x[key] for x in list_dict]


def dill_serialized_execute_func(f, *input_tuple):
    """Load the function that has been serialized by dill, and subsequently execute it.

    Args:
        f: function
            The function that has been serialized by dill.
        *input_tuple:

    Returns:
        The output of 'f'.
    """
    # noinspection PyBroadException
    try:
        import dill
        f = dill.loads(f)
        return f(*input_tuple)
    except Exception:
        return None


class ProcessPool(object):
    def __init__(self, num_processes: int = None, interval_sec: int = 0):
        """ A pool of processing.

        Args:
            num_processes: int
                Number of concurrently executing processes.
            interval_sec: int
                Interval seconds between tasks. (Invalid when use async_pool)
        """
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.interval_sec = interval_sec

    def map(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, target, dynamic_param_list: List[tuple], static_param: tuple = None, **kwargs):
        """ Run 'target' function.

        Args:
            target: function
                Function like func(*static_param, *dynamic_param)
            dynamic_param_list: List[tuple]
            static_param: tuple

        Returns:
            List of result in every target function. (drop the output which is None)
        """
        static_param = static_param if static_param else tuple()
        if dynamic_param_list is None or len(dynamic_param_list) == 0:
            return None

        if self.num_processes > 0:
            res_list = self._run_async_pool(
                target,
                dynamic_param_list=dynamic_param_list,
                static_param=static_param,
                chunksize=kwargs.get('chunksize')
            )
        else:
            res_list = self._run_sync(target, dynamic_param_list=dynamic_param_list, static_param=static_param)

        return res_list

    # --------------------------- Private Functions ----------------------------------
    def _run_sync(self, target, dynamic_param_list: List[tuple], static_param: tuple):
        """ Run 'target' in main process for each task.

        Args:
            target:
            dynamic_param_list:
            static_param:

        Returns:

        """
        result_list = list()
        for dp in dynamic_param_list:
            out = target(*static_param, *dp)
            if out is not None:
                result_list.append(out)
        return result_list

    def _run_async_pool(self, target, dynamic_param_list: List[tuple], static_param: tuple, chunksize: int):
        """ Run 'target' in multiprocessing using pool.

        Args:
            target:
            dynamic_param_list:
            static_param: tuple
            chunksize: int
                If None, then set it to num_tasks // num_processes

        Returns:
            List of result
        """
        import multiprocessing as mp
        import dill

        target = dill.dumps(target)
        param_list = [(target, *static_param, *dp) for dp in dynamic_param_list]
        if chunksize is None:
            chunksize = max(len(param_list) // self.num_processes, 1)
        with mp.Pool(self.num_processes) as pool:
            out_list = pool.starmap(
                func=dill_serialized_execute_func,
                iterable=param_list,
                chunksize=chunksize
            )
            res_list = []
            for res in out_list:
                if res is not None:
                    res_list.append(res)
            return res_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
