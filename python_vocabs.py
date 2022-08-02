
def create_value_vocab(data: dict) -> dict:
    """
    Creates a vocabulary of terminals from a file in the format of: {terminal: index}
    :param data:
    :return: dict of terminals and their indices.
    """
    vocab = {"": 0}
    counter = 1
    for rep, name in data.items():
        if rep:
            for val1, _, val2 in rep:
                if val1 not in vocab:
                    vocab[val1] = counter
                    counter += 1
                if val2 not in vocab:
                    vocab[val2] = counter
                    counter += 1
        if name not in vocab:
            vocab[name] = counter
            counter += 1
    return vocab


def create_path_vocab(data: dict) -> dict:
    vocab = {"": 0}
    counter = 1
    for rep, name in data.items():
        if rep:
            for val1, path, val2 in rep:
                if path not in vocab:
                    vocab[path] = counter
                    counter += 1
    return vocab


def create_tag_vocab(data: dict) -> dict:
    return {val: i for i, val in enumerate(list(set(data.values())))}
