from typing import List, Union


def levenshtein_distance(r: Union[str, List[str]], h: Union[str, List[str]]):
    """
    Calculate the Levenshtein distance between two lists or strings.
    """

    # Initialisation
    dold = list(range(len(h) + 1))
    dnew = list(0 for _ in range(len(h) + 1))

    # Computation
    for i in range(1, len(r) + 1):
        dnew[0] = i
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return dold[-1]


if __name__ == "__main__":
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("aaa", "aba") == 1
    assert levenshtein_distance("aba", "aaa") == 1
    assert levenshtein_distance("aa", "aaa") == 1
    assert levenshtein_distance("aaa", "aa") == 1
    assert levenshtein_distance("abc", "bcd") == 2
    assert levenshtein_distance(["hello", "world"], ["hello", "world", "!"]) == 1
    assert levenshtein_distance(["hello", "world"], ["world", "hello", "!"]) == 2
