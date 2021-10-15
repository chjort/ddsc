import tensorflow as tf
import transformers

from src.tokenizers import Tokenizer
from src.utils import load_json

DATA_PATH = "data/ddt/train.json"
SPECIAL_LABEL = "[SPECIAL]"

HF_MODEL_NAME = "Maltehb/danish-bert-botxo-ner-dane"
MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 2

#%% DATA
data = load_json(DATA_PATH)
samples = data["samples"]
label_set = data["meta"]["tokens_ner"]

# map label to numeric id
label_id_map = {l: i for i, l in enumerate(label_set)}
special_label_id = len(label_id_map)
label_id_map[SPECIAL_LABEL] = special_label_id
n_classes = len(label_id_map)

# get sentences, tokens, and labels
sents = [dic["sentence"] for dic in samples]
tokens = [dic["tokens"] for dic in samples]
labels = [dic["tokens_ner"] for dic in samples]
label_ids = [[label_id_map[l] for l in label] for label in labels]

# convert to tensors
sents = tf.constant(sents)
tokens_rag = tf.ragged.constant(tokens)
labels_rag = tf.ragged.constant(label_ids)

#%% TOKENIZER
hf_tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL_NAME)

tokenizer = Tokenizer.from_huggingface(
    hf_tokenizer,
    output_type=tf.int64,
)

#%% DATA PIPELINE
def vectorize(x, y):
    x, y = tokenizer.tokenize_words_with_align(x, y)
    x = tokenizer.truncate_sequence(x, MAX_SEQUENCE_LENGTH - 2)
    y = tokenizer.truncate_sequence(y, MAX_SEQUENCE_LENGTH - 2)
    x = tokenizer.concat_special_tokens(x)
    y = tokenizer.concat_special_label(y, special_label_id)
    return x, y


def mask_to_sample_weights(weight_mask):
    weight_mask = tf.cast(weight_mask, tf.float32)
    n_samples = tf.cast(tf.size(weight_mask), tf.float32)
    sample_weights = weight_mask / tf.reduce_sum(weight_mask) * n_samples
    return sample_weights


def get_attention_mask(x, y):
    mask = tf.not_equal(x, tokenizer.pad_token_id)
    mask = tf.cast(mask, tf.int32)
    sample_weights = mask_to_sample_weights(mask)
    return (x, mask), y, sample_weights


train_td = tf.data.Dataset.from_tensor_slices((tokens_rag, labels_rag))
train_td = train_td.map(vectorize)
train_td = train_td.padded_batch(
    batch_size=BATCH_SIZE,
    padded_shapes=([MAX_SEQUENCE_LENGTH], [MAX_SEQUENCE_LENGTH]),
    padding_values=(tokenizer.pad_token_id, special_label_id),
    drop_remainder=True,
)
train_td = train_td.map(get_attention_mask)
train_td = train_td.prefetch(-1)

#%% MODEL
input_token_ids = tf.keras.layers.Input(
    shape=[None], dtype=tokenizer.output_type, name="token_ids"
)
input_pad_mask = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name="pad_mask")

x = transformers.TFBertModel.from_pretrained(HF_MODEL_NAME, name="bert")(
    [input_token_ids, input_pad_mask]
)
x = x.last_hidden_state
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(n_classes)(x)

model = tf.keras.Model(inputs=[input_token_ids, input_pad_mask], outputs=x)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=optimizer,
    loss=loss,
)
model.summary()

#%%
model.fit(train_td, epochs=2)

#%%
# model = transformers.TFBertForTokenClassification.from_pretrained(HF_MODEL_NAME, name="bert")
# conf = transformers.AutoConfig.from_pretrained("Maltehb/danish-bert-botxo-ner-dane")
# conf.id2label
#
# train_td = tf.data.Dataset.from_tensor_slices((tokens_rag, labels_rag))
# train_td = train_td.map(vectorize)
#
# it = iter(train_td)
# x, y = next(it)
# mask = tf.ones_like(x)
# x = tf.expand_dims(x, 0)
# mask = tf.expand_dims(mask, 0)
#
# # (x, mask), y, w = next(it)
# x
# mask
#
# z = model.predict([x, mask])
# z = z.logits
# z
#
# pred_class_ids = z.argmax(-1)
# [conf.id2label[id_] for id_ in pred_class_ids[0]]
#
# list(zip(tokenizer.detokenize(x)[0].numpy(), [conf.id2label[id_] for id_ in pred_class_ids[0]]))
