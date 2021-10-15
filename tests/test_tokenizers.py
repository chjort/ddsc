import os
import shutil
import unittest

import tensorflow as tf

from src.tokenizers import Tokenizer

single_sentence_x = tf.constant("Rokoko-kommoder i Mathias Ortmanns stil.")
batch_sentence_x = tf.ragged.constant(
    [
        "Rokoko-kommoder i Mathias Ortmanns stil.",
        "Landsholdsspilleren, Lars Elstrup blev udvist.",
    ]
)

single_words_x = tf.constant(
    ["Rokoko-kommoder", "i", "Mathias", "Ortmanns", "stil", "."]
)
single_words_y = tf.constant(["B-MISC", "O", "B-PER", "I-PER", "O", "O"])
batch_words_x = tf.ragged.constant(
    [
        ["Rokoko-kommoder", "i", "Mathias", "Ortmanns", "stil", "."],
        ["Landsholdsspilleren", ",", "Lars", "Elstrup", "blev", "udvist", "."],
    ]
)
batch_words_y = tf.ragged.constant(
    [
        ["B-MISC", "O", "B-PER", "I-PER", "O", "O"],
        ["O", "O", "B-PER", "I-PER", "O", "O", "O"],
    ]
)

default_tokenizer = Tokenizer(vocab_size=1024, output_type=tf.int64)
default_tokenizer.adapt(batch_sentence_x)


class TestTokenizer(unittest.TestCase):
    def test_shape_tokenize_unbatched(self):
        xwt = default_tokenizer.tokenize(single_words_x, is_words=True)
        xst = default_tokenizer.tokenize(single_sentence_x, is_words=False)
        self.assertEqual(xwt.shape, xst.shape)

    def test_shape_tokenize_batched(self):
        xwt = default_tokenizer.tokenize(batch_words_x, is_words=True)
        xst = default_tokenizer.tokenize(batch_sentence_x, is_words=False)
        tf.assert_equal(xwt.bounding_shape(), xst.bounding_shape())

    def test_tokenize_unbatched_fail_sent_is_words(self):
        fail_fn = lambda: default_tokenizer.tokenize(single_sentence_x, is_words=True)
        self.assertRaises(ValueError, fail_fn)

    def test_tokenize_batched_fail_rank2_not_words(self):
        fail_fn = lambda: default_tokenizer.tokenize(batch_words_x, is_words=False)
        self.assertRaises(ValueError, fail_fn)

    def test_tokenize_batched_fail_rank3(self):
        x = tf.stack([batch_words_x, batch_words_x], axis=0)
        fail_fn = lambda: default_tokenizer.tokenize(x, is_words=False)
        self.assertRaises(ValueError, fail_fn)

    def test_tokenize_words_with_align(self):
        x = single_words_x
        y = single_words_y
        xa, ya = default_tokenizer.tokenize_words_with_align(x, y)

        self.assertNotEqual(x.shape, xa.shape)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(xa.shape, ya.shape)

    def test_concat_special_token_ids_rank1(self):
        x = tf.constant([3541, 1926, 77], dtype=tf.int64)
        x = default_tokenizer.concat_special_tokens(x)

        target = tf.constant(
            [
                default_tokenizer.cls_token_id,
                3541,
                1926,
                77,
                default_tokenizer.sep_token_id,
            ],
            dtype=tf.int64,
        )
        tf.assert_equal(x, target)

    def test_concat_special_token_ids_rank2(self):
        x = tf.constant(
            [
                [10668, 77, 2265],
                [3541, 7118, 1926],
            ]
        )
        x = default_tokenizer.concat_special_tokens(x)

        target0 = tf.constant(
            [
                default_tokenizer.cls_token_id,
                10668,
                77,
                2265,
                default_tokenizer.sep_token_id,
            ]
        )
        target1 = tf.constant(
            [
                default_tokenizer.cls_token_id,
                3541,
                7118,
                1926,
                default_tokenizer.sep_token_id,
            ]
        )
        tf.assert_equal(x[0], target0)
        tf.assert_equal(x[1], target1)

    def test_concat_special_label_id_rank1(self):
        y = tf.constant([8, 3, 8], dtype=tf.int32)
        y = default_tokenizer.concat_special_label(y, 9)

        target = tf.constant([9, 8, 3, 8, 9], dtype=tf.int32)
        tf.assert_equal(y, target)

    def test_concat_special_label_id_rank2(self):
        y = tf.constant(
            [
                [2, 8, 8],
                [1, 0, 4],
            ],
            dtype=tf.int32,
        )
        y = default_tokenizer.concat_special_label(y, 9)

        target0 = tf.constant(
            [9, 2, 8, 8, 9],
            dtype=tf.int32,
        )
        target1 = tf.constant([9, 1, 0, 4, 9], dtype=tf.int32)
        tf.assert_equal(y[0], target0)
        tf.assert_equal(y[1], target1)

    def test_call_truncate_without_special_tokens(self):
        tokenizer = Tokenizer(
            vocab_size=1024,
            max_sequence_length=40,
            add_special_tokens=False,
            output_type=tf.int64,
        )
        tokenizer.adapt(batch_sentence_x)

        xt = tokenizer(batch_sentence_x)
        tokens_len = xt.bounding_shape()[1].numpy()
        self.assertEqual(tokenizer.max_sequence_length, tokens_len)

    def test_call_truncate_with_special_tokens(self):
        tokenizer = Tokenizer(
            vocab_size=1024,
            max_sequence_length=40,
            add_special_tokens=True,
            output_type=tf.int64,
        )
        tokenizer.adapt(batch_sentence_x)

        xt = tokenizer(batch_sentence_x)
        tokens_len = xt.bounding_shape()[1]

        sample1_start_token = xt[1][0].numpy()
        sample1_end_token = xt[1][-1].numpy()
        self.assertEqual(tokenizer.max_sequence_length, tokens_len)
        self.assertEqual(tokenizer.cls_token_id, sample1_start_token)
        self.assertEqual(tokenizer.sep_token_id, sample1_end_token)

    def test_call_pad(self):
        tokenizer = Tokenizer(
            vocab_size=1024,
            max_sequence_length=40,
            add_padding=True,
            output_type=tf.int64,
        )
        tokenizer.adapt(batch_sentence_x)

        xt = tokenizer(batch_sentence_x)
        tokens_len = xt.shape[1]
        sample0_last_element = xt[0][-1]

        self.assertEqual(tokenizer.max_sequence_length, tokens_len)
        self.assertEqual(tokenizer.pad_token_id, sample0_last_element)

    def test_save_load_tf(self):
        inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
        outputs = default_tokenizer(inputs)
        model = tf.keras.Model(inputs, outputs)

        save_path = "save/model"

        # save
        model.save(save_path, save_format="tf", include_optimizer=False)

        # load
        tf.keras.models.load_model(save_path, custom_objects={"Tokenizer": Tokenizer})

        # clean up
        shutil.rmtree("save")

    def test_save_load_h5(self):
        inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
        outputs = default_tokenizer(inputs)
        model = tf.keras.Model(inputs, outputs)

        save_path = "save/model.h5"

        # save
        model.save(save_path, save_format="h5", include_optimizer=False)

        # load
        tf.keras.models.load_model(save_path, custom_objects={"Tokenizer": Tokenizer})

        # clean up
        shutil.rmtree("save")

    def test_save_load_json(self):
        inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
        outputs = default_tokenizer(inputs)
        model = tf.keras.Model(inputs, outputs)

        os.makedirs("save")
        weights_path = "save/model_weights.h5"
        json_path = "save/model.json"

        # save
        model.save_weights(weights_path)
        with open(json_path, "w") as f:
            f.write(model.to_json())

        # load
        with open(json_path, "r") as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(
            model_json, custom_objects={"Tokenizer": Tokenizer}
        )
        model.load_weights(weights_path)

        # clean up
        shutil.rmtree("save")


if __name__ == "__main__":
    unittest.main()
