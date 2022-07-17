import json
import os
from typing import List, Tuple

from preprocessor import Preprocessor

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


if __name__ == '__main__':
    preprocessor = PythonPreprocessor()
    preprocessor.preprocess()
