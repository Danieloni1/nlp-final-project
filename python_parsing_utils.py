import ast
import json
from typing import Optional

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
    for func in list(funcs.keys())[:100]:
        try:
            parsed_func = ast.parse(func)
            # parsed[parsed_func] = funcs[func]
            parsed[parsed_func] = funcs[func]
        except Exception as e:
            bad.append(funcs[func])
    print(f"{len(bad)}/{len(funcs)} functions failed to parse")
    return parsed


def get_terminal_nodes(node: ast.AST) -> list:
    terminals = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            terminals.append(child)
    return terminals


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


def get_path(root: ast.AST, first: ast.AST, last: ast.AST) -> list:
    """
    Returns a list of nodes from first to last in the following format:
        [first, UP, node1, ..., closest_common_parent_node, DOWN, ...,  last]
    :param root:
    :param first:
    :param last:
    :return: path from first to last as a list of nodes.
    """
    path = [first]
    closest_common_parent = get_closest_common_parent(root, first, last)
    up_path = get_path_from_node_to_root(closest_common_parent, first)
    for node in up_path[1:]:
        path.append(UP)
        path.append(node)
    down_path = get_path_from_node_to_root(closest_common_parent, last)
    down_path.reverse()
    for node in down_path[1:]:
        path.append(DOWN)
        path.append(node)
    return path


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


def get_t_pairs(node: ast.AST):
    terminals = get_terminal_nodes(node)
    return [(a, b) for idx, a in enumerate(terminals) for b in terminals[idx + 1:]]


if __name__ == '__main__':
    parsed_funcs = parse_functions("./data/python.json")
    parsed_func = list(parsed_funcs.keys())[1]
    t_pairs = get_t_pairs(parsed_func)
    path_context = get_path_context(parsed_func, t_pairs[0][0], t_pairs[0][1])
