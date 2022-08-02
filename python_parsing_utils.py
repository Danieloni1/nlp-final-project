import ast
import json
from typing import Optional, Union

UP = "UP"
DOWN = "DOWN"


def parse_functions(file_path: str) -> dict:
    """
    Parses a file and returns a dictionary of functions.
    :param file_path: json file path containing functions as strings and their names.
    :return: dictionary of function ATSs and their names.
    """
    with open(file_path, "r") as f:
        funcs = json.load(f)
    parsed = {}
    bad = []
    for func in list(funcs.keys()):
        try:
            parsed_func = ast.parse(func)
            parsed[parsed_func] = funcs[func]
        except Exception as e:
            bad.append(funcs[func])
    print(f"{len(bad)}/{len(funcs)} functions failed to parse")
    return parsed


def get_terminal_nodes(node: ast.AST) -> tuple:
    terminals = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            terminals.append(child)
    return tuple(terminals)


def terminal_to_value(terminal: ast.Name) -> str:
    """Phi function"""
    return terminal.id


def get_non_terminal_nodes(node: ast.AST) -> list:
    non_terminals = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Name):
            non_terminals.append(child)
    return non_terminals


def non_terminal_to_value(non_terminal: ast.AST) -> str:
    return non_terminal.__class__.__name__


def get_path(root: ast.AST, first: ast.AST, last: ast.AST) -> tuple:
    """
    Returns a list of nodes from first to last in the following format:
        [first, UP, node1, ..., closest_common_parent_node, DOWN, ...,  last]
    :param root:
    :param first:
    :param last:
    :return: path from first to last as a list of nodes.
    """
    path = [first.__class__.__name__]
    closest_common_parent = get_closest_common_parent(root, first, last)
    up_path = get_path_from_node_to_root(closest_common_parent, first)
    for node in up_path[1:]:
        path.append(UP)
        path.append(node.__class__.__name__)
    down_path = get_path_from_node_to_root(closest_common_parent, last)
    down_path.reverse()
    for node in down_path[1:]:
        path.append(DOWN)
        path.append(node.__class__.__name__)
    return tuple(path)


def get_path_context(root: ast.AST, first: ast.Name, last: ast.Name) -> tuple:
    return terminal_to_value(first), get_path(root, first, last), terminal_to_value(last)


def get_closest_common_parent(root: ast.AST, first: ast.AST, second: ast.AST) -> Optional[ast.AST]:
    path_first = get_path_from_node_to_root(root, first)
    path_second = get_path_from_node_to_root(root, second)
    path_second.reverse()
    for node in path_first:
        if node in path_second:
            return node
    return None


def get_path_from_node_to_root(root: ast.AST, node: ast.AST):
    path = []
    set_parents(root)
    current_node = node
    while current_node is not root:
        path.append(current_node)
        if hasattr(current_node, "parent"):
            current_node = current_node.parent
    return path


def set_parents(root: ast.AST):
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def get_t_pairs(node: ast.AST, max_pairs: int = 100) -> list:
    terminals = get_terminal_nodes(node)
    return [(a, b) for idx, a in enumerate(terminals) for b in terminals[idx + 1:]][:max_pairs]


def get_input_representation(function: Union[str, ast.AST], rep_size: int = 10) -> tuple:
    function_ast = ast.parse(function) if isinstance(function, str) else function
    rep = set()
    fails = 0
    for pair in get_t_pairs(function_ast)[:rep_size]:
        try:
            rep.add(get_path_context(function_ast, pair[0], pair[1]))
        except Exception as e:
            fails += 1
    rep = pad_input_representation_if_necessary(tuple(rep), rep_size)
    if fails:
        print(f"{fails}/{rep_size} contexts failed to parse")
    return tuple(rep)


def pad_input_representation_if_necessary(input_representation: tuple, rep_size: int = 10) -> tuple:
    while len(input_representation) < rep_size:
        input_representation = input_representation + (("", "", ""),)
    return input_representation
