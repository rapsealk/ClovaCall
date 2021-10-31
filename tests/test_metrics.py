import tensorflow as tf

from las.tensorflow_impl.metrics import CharacterErrorRate


class MetricsTestCase(tf.test.TestCase):

    def setUp(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, oov_token='@')
        characters = [chr(i) for i in range(ord('가'), ord('힣')+1)]
        self.tokenizer.fit_on_texts(characters)

    def test_character_error_rate_metric(self):
        x = '가나다라'
        y = '가나다'

        print(x, y, character_error_rate(x, y))

        [x] = tokenizer.texts_to_sequences([x])
        [y] = tokenizer.texts_to_sequences([y])

        print(x)
        print(y)

        print(x, y, character_error_rate(x, y))


if __name__ == "__main__":
    tf.test.main()
