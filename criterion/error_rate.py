def character_error_rate(text: str, answer: str) -> float:
    """
    CER: Character Error Rate
    * CER = (S + D + I) / N = (S + D + I) / (S + D + C)
    - S: Number of Substitutions
    - D: Number of Deletions
    - I: Number of Insertions
    - N: Number of characters
    - C: Number of correctness
    """
    return levenshtein(text, answer) / max(len(text), len(answer))


def levenshtein(s1, s2, cost=None, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug=debug)

    if len(s2) == 0:
        return len(s1)

    if cost is None:
        cost = {}

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]


if __name__ == "__main__":
    pass
