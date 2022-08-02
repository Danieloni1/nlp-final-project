import json
import os
import time
from typing import List, Tuple

from preprocessor import Preprocessor
from python_parsing_utils import parse_functions, get_input_representation
from python_vocabs import create_value_vocab, create_path_vocab, create_tag_vocab

PYTHON_DATA_PATH = "data/raw_data/python"


class PythonPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__("python", PYTHON_DATA_PATH)
        self._indentation_level = 0
        self._inside_function = False
        self._current_function = ""

    def preprocess(self):
        functions = set()
        for filename in os.listdir(self.data_path):
            if filename.endswith(".py"):
                file_path = os.path.join(self.data_path, filename)
                funcs = set(self.extract_functions(file_path))
                functions.update(funcs)
        data_set = self.create_data_set(functions)
        data_set_json_object = json.dumps(data_set, indent=4)
        with open("data/python.json", "w") as f:
            f.write(str(data_set_json_object))

    def extract_functions(self, file_path: str) -> List[str]:
        with open(file_path, "r") as f:
            try:
                lines = f.readlines()
                functions = []
                for line in lines:
                    if line == "":
                        continue
                    if self._inside_function:
                        if self.get_indentation_level(line) <= self._indentation_level and line != "\n":
                            self.stop_extracting_function(functions)
                            continue
                        self._current_function += line[self._indentation_level:]
                    if line.strip(" ").startswith("def"):
                        self.start_extracting_function(line)
                functions = [function for function in functions if function.startswith("def ")]
                return functions
            except Exception as e:
                print(e)
                return []

    def preprocess_function(self, function: str) -> Tuple[str, str]:
        name = self.extract_name(function)
        return self.mask_name(function, name), name

    def create_data_set(self, functions: set) -> dict:
        data_set = {}
        for function in functions:
            function_body, function_name = self.preprocess_function(function)
            data_set[function_body] = function_name
        return data_set

    @staticmethod
    def get_indentation_level(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def start_extracting_function(self, line: str):
        self._indentation_level = self.get_indentation_level(line)
        self._current_function += line.lstrip(" ")
        self._inside_function = True

    def stop_extracting_function(self, functions: List[str]):
        self._inside_function = False
        functions.append(self._current_function)
        self._current_function = ""

    @staticmethod
    def mask_name(function: str, name: str) -> str:
        function_list = function.split("\n")
        function_signature = function_list[0].replace(name, "f")
        function_list[0] = function_signature
        return "\n".join(function_list)

    @staticmethod
    def extract_name(function: str) -> str:
        function_list = function.split("\n")
        return function_list[0].split("(")[0].split(" ")[1]


def prepare_data(file_path: str, rep_size=10) -> tuple:
    """
    Prepares data for training.
    :param rep_size: number of contexts to use for each instance.
    :param file_path: json file path containing functions as strings and their names.
    :return: tuple of:
        [0] dictionary of training data: functions as reps and their names.
        [1] dictionary of validation data: functions as reps and their names.
        [2] dictionary of test data: functions as reps and their names.
        [3] dictionary of values and their indices.
        [4] dictionary of paths and their indices.
        [5] dictionary of tags and their indices.
    """
    start = time.time()
    funcs = parse_functions(file_path)
    data = {}
    for func in list(funcs.keys()):
        rep = get_input_representation(func, rep_size)
        if rep:
            data[rep] = funcs[func]
    value_vocab = create_value_vocab(data)
    path_vocab = create_path_vocab(data)
    tag_vocab = create_tag_vocab(data)

    print("Preprocessing took {} seconds.".format(time.time() - start))
    training_data = dict(list(data.items())[:int(len(data) * 0.8)])
    validation_data = dict(list(data.items())[int(len(data) * 0.8):int(len(data) * 0.85)])
    test_data = dict(list(data.items())[int(len(data) * 0.85):])
    return training_data, validation_data, test_data, value_vocab, path_vocab, tag_vocab


if __name__ == '__main__':
    preprocessor = PythonPreprocessor()
    preprocessor.preprocess()
