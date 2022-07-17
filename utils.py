import re as re

PYTHON_FUNCTION_KEYWORD = "def"
JS_FUNCTION_KEYWORD = "function"
JS_FUNCTION_FORMS_REGEX = [
    re.compile(".*" + JS_FUNCTION_KEYWORD + ".*{.*"),
    re.compile(".*=.*(.*)\s*{.*"),
    re.compile("(.*)\s*=>\s*{.*")
]