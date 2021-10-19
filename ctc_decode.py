import numpy as np
import tensorflow as tf


def main():
    inf = float("inf")
    logits = tf.constant([[[0., -inf, -inf],
                           [-2.3, -inf, -0.1]],
                          [[-inf, -0.5, -inf],
                           [-inf, -inf, -0.1]],
                          [[-inf, -inf, -inf],
                           [-0.1, -inf, -2.3]]])
    seq_lens = tf.constant([2, 3])
    [decoded], neg_sum_logits = tf.nn.ctc_greedy_decoder(
        logits,
        seq_lens,
        merge_repeated=True,
        blank_index=1)

    print('logits:', logits.shape)
    print(f'decoded: {decoded}')
    print(f'decoded.dense_shape: {decoded.dense_shape}')
    print(f'decoded.indices: {decoded.indices}')
    print(f'decoded.values: {decoded.values}')


if __name__ == "__main__":
    main()
