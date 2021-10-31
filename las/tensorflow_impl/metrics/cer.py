import tensorflow as tf


class CharacterErrorRate(tf.keras.metrics.Metric):

    def __init__(self, name='character_error_rate', **kwargs):
        super(CharacterErrorRate, self).__init__(name=name, **kwargs)
        self.maximum_lengths = self.add_weight(name='maximum_lengths', initializer='zeros')
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


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
    print(f'levenshtein(s1={s1}, s2={s2})')
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug=debug)

    if len(s2) == 0:
        return len(s1)

    if cost is None:
        cost = {}

    def substitution_cost(c1, c2):
        print(f'substitution_cost(c1={c1}, c2={c2})')
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
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, oov_token='@')
    characters = [chr(i) for i in range(ord('가'), ord('힣')+1)]
    tokenizer.fit_on_texts(characters)

    x = '가나다라'
    y = '가나다'

    print(x, y, character_error_rate(x, y))

    [x] = tokenizer.texts_to_sequences([x])
    [y] = tokenizer.texts_to_sequences([y])

    print(x)
    print(y)

    print(x, y, character_error_rate(x, y))
