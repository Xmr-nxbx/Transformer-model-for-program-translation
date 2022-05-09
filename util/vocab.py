import json

from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]


def build_tokenizer(data_path, vocab_path, vocab_size=30000):
    with open(data_path, "r", encoding="utf-8") as f:
        training_set = json.load(f)
    data = []
    for data_dict in training_set:
        for k in data_dict.keys():
            data.append(data_dict[k])

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, continuing_subword_prefix='$$')
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save(vocab_path)


def tokenize_dataset(data_path, vocab_path):
    tokenizer = Tokenizer.from_file(vocab_path)
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    def get_token_sequence(text):
        output = tokenizer.encode(text.strip())
        token_list = output.tokens
        unk_places = [i for i in range(len(token_list)) if token_list[i] == '<unk>']
        unk_origin_place = [output.offsets[each] for each in unk_places]
        token_list = [
            text[unk_origin_place[unk_places.index(i)][0]: unk_origin_place[unk_places.index(i)][1]]
            if i in unk_places
            else token_list[i]
            for i in range(len(token_list))
        ]
        return token_list

    for each_data in dataset:
        for k in each_data.keys():
            each_data[k] = get_token_sequence(each_data[k])
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)


def vocab_token2id(vocab_path):
    tokenizer = Tokenizer.from_file(vocab_path)
    return tokenizer.get_vocab()


def map_token_to_id(data_path, vocab_path, lang1, lang2):
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    t2id = vocab_token2id(vocab_path)

    def map_func(token_seq):
        id_list = [
            t2id[w] if t2id.__contains__(w) else t2id["<unk>"] for w in token_seq
        ]
        return id_list

    for i in range(len(dataset)):
        dataset[i][lang1] = map_func(dataset[i][lang1 + "_raw"])
        dataset[i][lang2] = map_func(dataset[i][lang2 + "_raw"])

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
