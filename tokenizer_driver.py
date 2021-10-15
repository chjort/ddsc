import tensorflow as tf
import transformers

from src.tokenizers import Tokenizer, _align_tokens_and_labels, _batch_align_tokens_and_labels

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

#%%
# default_tokenizer = Tokenizer(vocab_size=1024, output_type=tf.int64)
# default_tokenizer.adapt(batch_sentence_x)

hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Maltehb/danish-bert-botxo-ner-dane"
)

tokenizer = Tokenizer.from_huggingface(
    hf_tokenizer,
    output_type=tf.string,
)

#%%
# x = single_words_x
# y = single_words_y

tokens, labels = tokenizer.tokenize_words_with_align(x, y)
print(*list(zip(tokens.numpy(), labels.numpy())), sep="\n")

xt = tokenizer._tokenizer.tokenize(x).merge_dims(-2, -1)
tokens, labels = _align_tokens_and_labels((xt, y))
print(*list(zip(tokens.numpy(), labels.numpy())), sep="\n")

x = batch_words_x
y = batch_words_y

tokens, labels = tokenizer.tokenize_words_with_align(x, y)
print(*list(zip(tokens[0].numpy(), labels[0].numpy())), sep="\n")
print(*list(zip(tokens[1].numpy(), labels[1].numpy())), sep="\n")

tokens, labels = _batch_align_tokens_and_labels((xt, y))
print(*list(zip(tokens[0].numpy(), labels[0].numpy())), sep="\n")
print(*list(zip(tokens[1].numpy(), labels[1].numpy())), sep="\n")

#%%
