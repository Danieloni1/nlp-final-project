import re as re

PYTHON_FUNCTION_KEYWORD = "def"
JS_FUNCTION_KEYWORD = "function"
JS_FUNCTION_FORMS_REGEX = [
    re.compile(JS_FUNCTION_KEYWORD + ".*{.*")
    # re.compile(".*" + JS_FUNCTION_KEYWORD + ".*{.*")
    # re.compile(".*=.*(.*)\s*{.*"),
    # re.compile("(.*)\s*=>\s*{.*")
]
JS_FUNCTION_ENDING_REGEX = re.compile(".*}.*")