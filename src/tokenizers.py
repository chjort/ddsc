import re

import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer
from tensorflow_text.python.ops.bert_tokenizer import BasicTokenizer
from tensorflow_text.tools.wordpiece_vocab import (
    wordpiece_tokenizer_learner_lib as wordpiece_learner,
)

_SET_TOKEN_ERROR_STRING = (
    "Can not set special tokens when tokenizer has already been adapted."
    " Use reset_state() method before setting special tokens."
)


def _create_table(
    keys, values=None, key_dtype=tf.string, value_dtype=tf.int64, num_oov_buckets=1
):

    if values is None:
        n = tf.size(keys, out_type=value_dtype)
        values = tf.range(n, dtype=value_dtype)

    kv = tf.lookup.KeyValueTensorInitializer(
        keys=keys,
        values=values,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
    )
    table = tf.lookup.StaticVocabularyTable(
        initializer=kv, num_oov_buckets=num_oov_buckets
    )
    return table


def _align_tokens_and_labels(tokens_labels: tuple):
    tokens = tokens_labels[0]
    labels = tokens_labels[1]

    tokens_word_id = tokens.nested_value_rowids()
    labels = tf.gather(labels, tokens_word_id)
    labels = tf.squeeze(labels, axis=0)
    tokens = tokens.flat_values
    labels = tf.ensure_shape(labels, tokens.shape)
    return tokens, labels


def _batch_align_tokens_and_labels(tokens_labels: tuple):
    tokens = tokens_labels[0]
    labels = tokens_labels[1]

    tokens, labels = tf.map_fn(
        _align_tokens_and_labels,
        (tokens, labels),
        fn_output_signature=(
            tf.RaggedTensorSpec(shape=[None], dtype=tokens.dtype),
            tf.RaggedTensorSpec(shape=[None], dtype=labels.dtype),
        ),
    )

    return tokens, labels


def _pad_ragged_sequence(sequence, max_seq_length, pad_value=0):
    # Verify that everything is a RaggedTensor
    if not isinstance(sequence, tf.RaggedTensor):
        raise TypeError("Expecting a `RaggedTensor`, instead found: " + str(sequence))

    # Flatten down to `merge_axis`
    sequence = sequence.merge_dims(1, -1) if sequence.ragged_rank > 1 else sequence

    # Pad to fixed Tensor
    target_shape = tf.cast([-1, max_seq_length], tf.int64)
    padded_input = sequence.to_tensor(shape=target_shape, default_value=pad_value)

    # Get padded input mask
    input_mask = tf.ones_like(sequence, tf.int64)
    padded_input_mask = input_mask.to_tensor(shape=target_shape)

    return padded_input, padded_input_mask


class Tokenizer(PreprocessingLayer):
    def __init__(
        self,
        vocab_size=None,
        vocabulary=None,
        max_sequence_length=None,
        add_padding=False,
        add_special_tokens=False,
        lower_case=False,
        split_unknown_characters=False,
        inputs_as_words=False,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        output_type=tf.int64,
        name=None,
        **kwargs,
    ):
        if vocabulary is None and vocab_size is None:
            raise ValueError(
                "`vocab_size` must be specified if `vocabulary` is not given."
            )

        if add_padding and max_sequence_length is None:
            raise ValueError(
                "`max_sequence_length` must be specified if `add_padding` is True."
            )

        super(Tokenizer, self).__init__(name=name, **kwargs)
        self._stateful = True
        self._streaming = True
        self._from_vocabulary = False

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.add_padding = add_padding
        self.add_special_tokens = add_special_tokens
        self.lower_case = lower_case
        self.split_unknown_characters = split_unknown_characters
        self.inputs_as_words = inputs_as_words
        self.output_type = output_type

        self._suffix_indicator = "##"
        self._keep_whitespace = False
        self._normalization_form = None
        self._preserve_unused_token = False

        self._pad_token = pad_token
        self._cls_token = cls_token
        self._sep_token = sep_token
        self._unk_token = unk_token
        self._mask_token = mask_token

        self._text_splitter = BasicTokenizer(
            lower_case=self.lower_case,
            keep_whitespace=self._keep_whitespace,
            normalization_form=self._normalization_form,
            preserve_unused_token=self._preserve_unused_token,
        )
        self._tokenizer = None

        if inputs_as_words:
            self.input_spec = tf.keras.layers.InputSpec(shape=(None, None))
        else:
            self.input_spec = tf.keras.layers.InputSpec(shape=(None,))

        self._truncate_length = self.max_sequence_length
        if self.max_sequence_length is not None and add_special_tokens:
            n_special_tokens = 2
            if self._cls_token is None:
                n_special_tokens -= 1
            if self._sep_token is None:
                n_special_tokens -= 1

            self._truncate_length -= n_special_tokens

        if vocabulary is not None:
            if isinstance(vocabulary, (str, tf.lookup.StaticVocabularyTable)):
                vocab = vocabulary
            else:
                vocab = _create_table(vocabulary)

            self._configure_tokenizer(vocab)
            self._from_vocabulary = True

    def call(self, inputs, *args, **kwargs):
        tokens = self.tokenize(inputs, is_words=self.inputs_as_words)

        if self.max_sequence_length is not None:
            tokens = self.truncate_sequence(tokens, self._truncate_length)

        if self.add_special_tokens:
            tokens = self.concat_special_tokens(tokens)

        if self.add_padding:
            if tf.as_dtype(self.output_type).is_integer:
                pad_value = self.pad_token_id
            else:
                pad_value = self.pad_token

            tokens, _ = self.pad_sequence(
                tokens,
                max_seq_len=self.max_sequence_length,
                pad_value=pad_value,
            )

        return tokens

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def cls_token(self):
        return self._cls_token

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def unk_token(self):
        return self._unk_token

    @property
    def mask_token(self):
        return self._mask_token

    @pad_token.setter
    def pad_token(self, value):
        if self.is_adapted:
            raise ValueError(_SET_TOKEN_ERROR_STRING)
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        if self.is_adapted:
            raise ValueError(_SET_TOKEN_ERROR_STRING)
        self._cls_token = value

    @sep_token.setter
    def sep_token(self, value):
        if self.is_adapted:
            raise ValueError(_SET_TOKEN_ERROR_STRING)
        self._sep_token = value

    @unk_token.setter
    def unk_token(self, value):
        if self.is_adapted:
            raise ValueError(_SET_TOKEN_ERROR_STRING)
        self._unk_token = value

    @mask_token.setter
    def mask_token(self, value):
        if self.is_adapted:
            raise ValueError(_SET_TOKEN_ERROR_STRING)
        self._mask_token = value

    def tokenize(self, text, is_words=False):
        self._validate_is_adapted()
        ndims_text = text.shape.ndims

        tokens = self._tokenizer.tokenize(text)

        if ndims_text == 0 and not is_words:  # single sentence
            tokens = tokens.merge_dims(-3, -1)
        elif ndims_text == 0 and is_words:
            raise ValueError(
                "Input must have rank 1 or 2 when is_words=True, found rank {}.".format(
                    ndims_text
                )
            )
        elif ndims_text == 1 and not is_words:  # batch of sentences
            tokens = tokens.merge_dims(-2, -1)
        elif ndims_text == 1 and is_words:  # single list of words
            tokens = tokens.merge_dims(-3, -1)
        elif ndims_text == 2 and is_words:  # batch of lists of words
            tokens = tokens.merge_dims(-3, -1)
        elif ndims_text == 2 and not is_words:
            raise ValueError(
                "Input must have rank less than 1, found rank {}. Set is_words=True if inputs are words.".format(
                    ndims_text
                )
            )
        else:
            raise ValueError(
                "Input must have rank less than 2, found rank {}.".format(ndims_text)
            )

        return tokens

    def tokenize_with_offsets(self, text):
        self._validate_is_adapted()
        ndims_text = text.shape.ndims

        if ndims_text == 0:  # single sentence
            tokens, start, end = self._tokenizer.tokenize_with_offsets(text)
            tokens = tokens.merge_dims(-3, -1)
            start = start[0]
            end = end[0]
        elif ndims_text == 1:  # batch of sentences
            tokens, start, end = self._tokenizer.tokenize_with_offsets(text)
            tokens = tokens.merge_dims(-2, -1)
        else:
            raise ValueError(
                "Input must have rank 0 or 1, found rank {}.".format(ndims_text)
            )

        # start = tf.reduce_min(start, axis=-1)
        # end = tf.reduce_max(end, axis=-1)
        start = start.merge_dims(-2, -1)
        end = end.merge_dims(-2, -1)

        return tokens, start, end

    def tokenize_words_with_align(self, x, y):
        self._validate_is_adapted()
        ndims_x = x.shape.ndims
        if ndims_x == 1 or ndims_x == 2:
            x = self._tokenizer.tokenize(x).merge_dims(-2, -1)
        else:
            raise ValueError("Input must have rank 1 or 2, found rank {}.".format(ndims_x))

        if ndims_x == 1:
            x, y = _align_tokens_and_labels((x, y))
        else:
            x, y = _batch_align_tokens_and_labels((x, y))

        return x, y

    def get_vocab_and_ids(self):
        self._validate_is_adapted()
        vocab, ids = self._tokenizer._wordpiece_tokenizer._get_vocab_and_ids()
        return vocab, ids

    def ids_to_tokens(self, token_ids):
        vocab, ids = self.get_vocab_and_ids()

        first_is_zero = tf.equal(ids[0], 0)
        steps = ids[1:] - ids[:-1]
        all_one_step = tf.reduce_all(tf.equal(steps, 1))
        check = tf.Assert(
            first_is_zero & all_one_step,
            data=[
                (
                    "`detokenize` only works with vocabulary tables where the "
                    "indices are dense on the interval `[0, vocab_size)`"
                )
            ],
        )
        with tf.control_dependencies([check]):
            # Limit the OOV buckets to a single index.
            token_ids = tf.minimum(
                token_ids,
                tf.cast(tf.size(vocab), token_ids.dtype),
            )

        # Add the unknown token at that index.
        vocab = tf.concat([vocab, [self.unk_token]], axis=0)

        return tf.gather(vocab, token_ids)

    def combine_wordpieces(self, tokens):
        ndims_tokens = tokens.shape.ndims
        if ndims_tokens == 1:
            tokens = tf.expand_dims(tokens, 0)

        # Ensure the input is Ragged.
        if not isinstance(tokens, tf.RaggedTensor):
            tokens = tf.RaggedTensor.from_tensor(tokens)

        # Join the tokens along the last axis.
        words = tf.strings.reduce_join(tokens, axis=-1, separator=" ")

        # Collapse " ##" in all strings to make words.
        words = tf.strings.regex_replace(
            words, " " + re.escape(self._suffix_indicator), ""
        )

        # Strip leading and trailing spaces.
        words = tf.strings.regex_replace(words, "^ +| +$", "")

        # Split on spaces so the last axis is "words".
        words = tf.strings.split(words, sep=" ")

        if ndims_tokens == 1:
            words = words[0]

        return words

    def detokenize(self, token_ids):
        tokens = self.ids_to_tokens(token_ids)
        tokens = self.combine_wordpieces(tokens)
        return tokens

    def concat_special_tokens(self, x):
        self._validate_is_adapted()

        if tf.as_dtype(self.output_type).is_integer:
            cls_token = self.cls_token_id
            sep_token = self.sep_token_id
        else:
            cls_token = self.cls_token
            sep_token = self.sep_token

        if cls_token is not None:
            cls_token = [cls_token]
        else:
            cls_token = None

        if sep_token is not None:
            sep_token = [sep_token]
        else:
            sep_token = None

        x = text.pad_along_dimension(
            x, axis=-1, left_pad=cls_token, right_pad=sep_token
        )

        return x

    def concat_special_label(self, y, special_label):

        if self.cls_token is not None:
            left_value = [special_label]
        else:
            left_value = None

        if self.sep_token is not None:
            right_value = [special_label]
        else:
            right_value = None

        y = text.pad_along_dimension(
            y, axis=-1, left_pad=left_value, right_pad=right_value
        )

        return y

    @staticmethod
    def truncate_sequence(x, max_seq_len):
        ndims = x.shape.ndims
        if ndims == 1:
            x = x[:max_seq_len]
        elif ndims == 2:
            x = x[:, :max_seq_len]
        else:
            raise ValueError(
                "Input must have rank 1 or 2, found rank {}.".format(ndims)
            )
        return x

    @staticmethod
    def pad_sequence(x, max_seq_len, pad_value):
        x, pad_mask = _pad_ragged_sequence(
            x,
            max_seq_length=max_seq_len,
            pad_value=pad_value,
        )
        return x, pad_mask

    def adapt(self, data):
        if not self._is_compiled:
            self.compile()

        if self.built:
            self.reset_state()

        if isinstance(data, tf.data.Dataset):
            tokens = data.map(self._text_splitter.tokenize)
            self._adapt_maybe_build(next(iter(tokens)))
        else:
            tokens = self._text_splitter.tokenize(data)
            self._adapt_maybe_build(tokens)

        word_counts = wordpiece_learner.count_words(tokens)

        reserved_tokens = [
            self.pad_token,
            self.cls_token,
            self.sep_token,
            self.unk_token,
            self.mask_token,
        ]
        reserved_tokens = list(filter(None, reserved_tokens))

        vocab = wordpiece_learner.learn(
            word_counts=word_counts,
            vocab_size=self.vocab_size,
            reserved_tokens=reserved_tokens,
            joiner=self._suffix_indicator,
        )
        vocab = _create_table(vocab)
        self._configure_tokenizer(vocab)

    def reset_state(self):
        self._from_vocabulary = False
        self._tokenizer = None

    def _configure_tokenizer(self, vocab):
        self._tokenizer = text.BertTokenizer(
            vocab_lookup_table=vocab,
            suffix_indicator=self._suffix_indicator,
            max_bytes_per_word=100,
            max_chars_per_token=None,
            token_out_type=self.output_type,
            unknown_token=self.unk_token,
            split_unknown_characters=self.split_unknown_characters,
            lower_case=self.lower_case,
            keep_whitespace=self._keep_whitespace,
            normalization_form=self._normalization_form,
            preserve_unused_token=self._preserve_unused_token,
            basic_tokenizer_class=BasicTokenizer,
        )
        self._vocab_table = self._tokenizer._wordpiece_tokenizer._vocab_lookup_table
        self.vocab_size = (
            self._vocab_table.size().numpy() - 1
        )  # -1 to ignore OOV bucket

        self._is_adapted = True

        self.pad_token_id = (
            self._vocab_lookup(self.pad_token) if self.pad_token is not None else None
        )
        self.cls_token_id = (
            self._vocab_lookup(self.cls_token) if self.cls_token is not None else None
        )
        self.sep_token_id = (
            self._vocab_lookup(self.sep_token) if self.sep_token is not None else None
        )
        self.unk_token_id = (
            self._vocab_lookup(self.unk_token) if self.unk_token is not None else None
        )
        self.mask_token_id = (
            self._vocab_lookup(self.mask_token) if self.mask_token is not None else None
        )

    def _vocab_lookup(self, x):
        self._validate_is_adapted()
        x = tf.convert_to_tensor(x, dtype=tf.string)
        return self._vocab_table.lookup(x).numpy()

    def _validate_is_adapted(self):
        if not self._is_adapted:
            raise AssertionError(
                "Tokenizer has not been adapted. "
                "Initialize tokenizer with a vocabulary or call the adapt method."
            )

    @classmethod
    def from_huggingface(cls, tokenizer, **kwargs):
        vocab = tokenizer.get_vocab()
        keys, values = zip(*vocab.items())
        table = _create_table(keys, values)

        config = tokenizer.init_kwargs
        return cls(
            vocabulary=table,
            lower_case=config["do_lower_case"],
            pad_token=tokenizer.pad_token,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            unk_token=config["unk_token"],
            mask_token=tokenizer.mask_token,
            **kwargs,
        )

    def get_config(self):
        vocabulary, _ = self.get_vocab_and_ids()
        vocabulary = vocabulary.numpy().tolist()
        vocabulary = [w.decode("utf-8") for w in vocabulary]
        config = {
            "vocab_size": self.vocab_size,
            "vocabulary": vocabulary,
            "max_sequence_length": self.max_sequence_length,
            "add_padding": self.add_padding,
            "add_special_tokens": self.add_special_tokens,
            "lower_case": self.lower_case,
            "split_unknown_characters": self.split_unknown_characters,
            "inputs_as_words": self.inputs_as_words,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token,
            "output_type": tf.as_dtype(self.output_type).name,
        }
        base_config = super(Tokenizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""
Use case 1: args: (x)
    "sentence" -> tokenize -> ["token1", "##token2", ..., "tokenN"]

Use case 2: args: (x)
                                                    [
                                                        ["token1", "##token2", ..., "tokenN"],
                                                        ["token1", "##token2", ..., "tokenN"],
    ["sent1", "sent2", ..., "sentN"] -> tokenize ->     ...,
                                                        ["token1", "##token2", ..., "tokenN"]
                                                    ]

Use case 2: args: (x, is_words)
    ["word1", "word2", ..., "wordN"] -> tokenize -> ["token1", "##token2", ..., "tokenN"]

Use case 4: args: (x, is_words)
    [                                                    [
        ["word1", "word2", ..., "wordN"],                    ["token1", "##token2", ..., "tokenN"],
        ["word1", "word2", ..., "wordN"],                    ["token1", "##token2", ..., "tokenN"],
        ...,                              -> tokenize ->     ...,
        ["word1", "word2", ..., "wordN"],                    ["token1", "##token2", ..., "tokenN"],
    ]                                                    ]

"""
