from typing import List


class Preprocessor:
    def __init__(self, name: str, path: str):
        self.name = name
        self.data_path = path

    def preprocess(self):
        """
        Preprocess all functions in self.data_path, create data set and store it json file.
        :return:
        """
        raise NotImplementedError()

    def extract_functions(self, file_path: str) -> List[str]:
        """
        Extract all functions (including signature) from file
        :param file_path:
        :return: list of functions
        """
        raise NotImplementedError()

    def preprocess_function(self, function: str) -> tuple:
        """
        Preprocess function body.
        :param function: raw function including signature and body
        :return: tuple of (function_body, function_name)
        """
        raise NotImplementedError()

    def create_data_set(self, functions: list) -> dict:
        """
        Split each functions into sample and label.
        :param functions: list of raw functions
        :return: dict of (function_body, function_name)
        """
        raise NotImplementedError()
