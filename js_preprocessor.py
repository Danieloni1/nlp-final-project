from typing import List

from preprocessor_interface import Preprocessor
from utils import *

JS_PATH = "./data/raw_data/js"


class jsPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__("js", JS_PATH)

    def preprocess(self):
        """
    Preprocess all functions in self.data_path, create data set and store it json file.
    :return:
    """
        """ Calls create_data_set 3 times for train, validation and test"""
        raise NotImplementedError()

    def extract_functions(self, file_path: str) -> List[str]:
        """
    Extract all functions (including signature) from file
    :param file_path:
    :return: list of functions
    """
        f = open(file_path, "r")
        lines = f.readlines()
        encapsulation_count = 0
        current_func = ""
        functions_list = []
        for line in lines:
            if any(regex.match(line) for regex in JS_FUNCTION_FORMS_REGEX):
                encapsulation_count += line.count('{')
            if encapsulation_count > 0:
                current_func += line
                if '}' in line:
                    encapsulation_count -= line.count('}')
            if encapsulation_count <= 0:
                if current_func != "":
                    functions_list.append(current_func)
                    current_func = ""
        return functions_list
        
    def preprocess_function(self, function: str) -> tuple:
        """
    Preprocess function body.
    :param function: raw function including signature and body
    :return: tuple of (function_body, function_name)
    """
        """ 2 possibilities: variable named functions, and classic function+name """
        raise NotImplementedError()

    def create_data_set(self, functions: list) -> dict:
        """
    Split each functions into sample and label.
    :param functions: list of raw functions
    :return: dict of (function_body, function_name)
    """
        """ Calls extract_functions and then iterates over output with preprocess_function"""
        raise NotImplementedError()
