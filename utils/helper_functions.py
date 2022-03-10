import inspect
import os
import pkgutil
import importlib
import sys
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.qpg.sacfd import SACfD

def get_kwargs(function):
    args = []
    for c in inspect.getmro(function):
        args.extend(list(inspect.signature(c).parameters.keys()))
    return list(set(args))


def subdict(org_dict, keys):
    return {k: org_dict[k] for k in keys if k in org_dict}


def intersect(listA, listB):
    return list(set(listA).intersection(set(listB)))


def get_relevant_kwargs(function, dict):
    func_kwargs = get_kwargs(function)
    intersection_kwargs = intersect(dict, func_kwargs)
    return subdict(dict, intersection_kwargs)

def get_dir():
    return os.path.dirname(os.path.abspath(__file__))