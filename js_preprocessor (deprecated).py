import json
import os
from typing import List

from preprocessor import Preprocessor
from utils import *

JS_PATH = "data/raw_data/js"


class JsPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__("js", JS_PATH)
        self._indentation_level = 0
        self._inside_function = False
        self._current_function = ""

    def preprocess(self):
        """
    Preprocess all functions in self.data_path, create data set and store it json file.
    :return:
    """
        """ Calls create_data_set 3 times for train, validation and test"""
        functions = set()
        for filename in os.listdir(self.data_path):
            if filename.endswith(".js"):
                file_path = os.path.join(self.data_path, filename)
                funcs = set(self.extract_functions(file_path))
                functions.update(funcs)
        data_set = self.create_data_set(functions)
        data_set_json_object = json.dumps(data_set, indent=4)
        with open("data/javascript.json", "w") as f:
            f.write(str(data_set_json_object))

    def extract_functions(self, file_path: str) -> List[str]:
        """
    Extract all functions (including signature) from file
    :param file_path:
    :return: list of functions
    """
        with open(file_path, "r") as f:
            try:
                lines = f.readlines()
                functions = []
                for line in lines:
                    if line == "":
                        continue
                    if self._inside_function:
                        if self._indentation_level == 0 and JS_FUNCTION_ENDING_REGEX.match(line):
                            self._current_function += line
                            self.stop_extracting_function(functions)
                            continue
                        self._current_function += line
                    if any(regex.match(line) for regex in JS_FUNCTION_FORMS_REGEX):
                        self.start_extracting_function(line)
                functions = [function for function in functions if any(regex.match(function) for regex in JS_FUNCTION_FORMS_REGEX)]
                return functions
            except Exception as e:
                print(e)
                return []

    def preprocess_function(self, function: str) -> tuple:
        """
    Preprocess function body.
    :param function: raw function including signature and body
    :return: tuple of (function_body, function_name)
    """
        name = self.extract_name(function)
        if name == "":
            return "", ""
        return self.mask_name(function, name), name

    def create_data_set(self, functions: list) -> dict:
        """
    Split each functions into sample and label.
    :param functions: list of raw functions
    :return: dict of (function_body, function_name)
    """
        """ Calls extract_functions and then iterates over output with preprocess_function"""
        data_set = {}
        for function in functions:
            function_body, function_name = self.preprocess_function(function)
            if function_body == "" or function_name == "":
                continue
            data_set[function_body] = function_name
        return data_set

    def start_extracting_function(self, line: str):
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
        try:
            m = re.search("function .*(.*)\s*\n", function).group(0)
            return m.split("function ")[1].split("(")[0].strip()
            # function_list = function.split("\n")[0].split("function")
            # return function_list[0][:-8] + function_list[1].split("(")[0].split(" ")[1]
        except AttributeError:
            return ""


if __name__ == '__main__':
    preprocessor = JsPreprocessor()
    preprocessor.preprocess()

# Try running this and see that the functions_list is filled with garbage. Need to adjust the extrat_functions method.
# Maybe use the _inside_function attribute Nitzan added.
# Maybe also decide whether or not we want to handle different types of function declarations other than the classic one.