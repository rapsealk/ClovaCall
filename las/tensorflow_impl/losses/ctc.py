import tensorflow as tf
import numpy as np


class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, label_length, logits_time_major=False, blank_index=-1, name='ctc_loss'):
        """
        `labels`: [batch_size, max_label_seq_length]
        `logits`: [batch_size, frames, num_labels]
        `logits_time_major`: [time, batch, logits] if True else [batch, time, logits]
        """
        super(CTCLoss, self).__init__(name=name)
        self.label_length = label_length
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.sparse.from_dense(y_true)
        logit_length = tf.fill(dims=[y_pred.shape[0]], value=y_pred.shape[1])
        logit_length = tf.cast(logit_length, dtype=tf.int32)
        loss = tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=self.label_length, logit_length=logit_length,
                              logits_time_major=self.logits_time_major, blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)


class SimpleSparseTensor:

    def __new__(self, x):
        """Create a very simple SparseTensor with dimensions (batch, time).
        Args:
            x: a list of lists of type int
        Returns:
            x_ix and x_val, the indices and values of the SparseTensor<2>.
        """
        x_ix = []
        x_val = []
        for batch_i, batch in enumerate(x):
            for time, val in enumerate(batch):
                x_ix.append([batch_i, time])
                x_val.append(val)
        x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
        x_ix = tf.constant(x_ix, tf.int64)
        x_val = tf.constant(x_val, tf.int32)
        x_shape = tf.constant(x_shape, tf.int64)

        return tf.SparseTensor(x_ix, x_val, x_shape)


def main():
    depth = 6   # max_time_steps(7) - 1

    targets_0 = [0, 1, 2, 1, 0]
    # loss_log_prob_0 = -3.34211
    input_prob_matrix_0 = np.asarray(
        [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

    targets_1 = [0, 1, 1, 0]
    # loss_log_prob_1 = -5.42262
    input_prob_matrix_1 = np.asarray(
        [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
         [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)
    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)

    inputs = [np.vstack([input_log_prob_matrix_0[t, :],
                         input_log_prob_matrix_1[t, :]])
              for t in range(5)] + 2 * [np.nan*np.ones((2, depth), np.float32)]
    inputs = np.asarray(inputs, dtype=np.float32)

    labels = SimpleSparseTensor([targets_0, targets_1])

    seq_lens = np.array([5, 5], dtype=np.int32)

    inputs = np.transpose(inputs, axes=(1, 0, 2))

    loss = tf.nn.ctc_loss(labels=labels, logits=inputs, label_length=10, logit_length=seq_lens, blank_index=10, logits_time_major=False)

    print('loss:', loss)


if __name__ == "__main__":
    main()
