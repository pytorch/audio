from collections.abc import Sequence


def levenshtein_distance(r: Sequence, h: Sequence) -> int:
    """
    Calculate the Levenshtein distance between two sequences.

    The function computes an edit distance allowing deletion, insertion and substitution.
    The result is an integer. Users may want to normalize by the length of the reference.

    This can be used to compute the edit distance for instance between two strings,
    or two list of words. Note that, if a string and a list of words is provided, the distance
    will be computed between the "string" sequence, and the "list of words" sequence.

    Args:
        r (Sequence): the reference sequence to compare.
        h (Sequence): the hypothesis sequence (e.g. the predicted sequence) to compare.
    Returns:
        int: The distance between the reference and the hypothesis.
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
