import os

from util import format, vocab, property, gen_ref

vocab_size = 10000
keyword = ['train', 'valid', 'test']


def step1_formatting(dataset_path):
    print("step1: Prepare and format the dataset")
    return format.get_formatted_data(dataset_path)


def step2_build_vocab_and_tokenize_dataset(dataset_path):
    print("step2: Create vocabulary and split tokens")
    dataset_saving_dir = os.path.join(dataset_path, "data")
    vocab_file_path = os.path.join(dataset_path, "vocab.json")
    all_dataset_dict = {
        k: os.path.join(dataset_saving_dir, "%s.json" % k)
        for k in keyword}
    vocab.build_tokenizer(all_dataset_dict['train'], vocab_file_path, vocab_size)

    for v in all_dataset_dict.values():
        vocab.tokenize_dataset(v, vocab_file_path)


def step3_token_to_id(dataset_path, lang1, lang2):
    print("step3: Convert token to id")
    dataset_saving_dir = os.path.join(dataset_path, "data")
    vocab_file_path = os.path.join(dataset_path, "vocab.json")
    dataset_saving_list = [os.path.join(dataset_saving_dir, "%s.json" % k) for k in keyword]
    for p in dataset_saving_list:
        vocab.map_token_to_id(p, vocab_file_path, lang1, lang2)


def step4_show_property(dataset_path, lang1, lang2):
    print("step4: Show training set statistics")
    training_set = os.path.join(dataset_path, "data/train.json")
    property.show_dataset_property(training_set, lang1, lang2)


def step5_gen_ref(dataset_path, lang1, lang2):
    print("step5: Generate reference")
    test_path = os.path.join(dataset_path, "data/test.json")
    gen_ref.generate_reference(dataset_path, test_path, lang1, lang2)


def main():
    path = "./data"
    dataset_collections = os.listdir(path)
    dataset_path = dataset_collections[0]
    if len(dataset_collections) != 1:
        print("Please select the dataset INDEX to be pre-processed:")
        for i, path in enumerate(dataset_collections):
            print("%d -> %s" % (i, path))
        select = input()
        assert type(select) == int and len(dataset_collections) > int(select) >= 0
        dataset_path = dataset_collections[int(select)]
    dataset_path = os.path.join(path, dataset_path)

    lang1, lang2 = step1_formatting(dataset_path)
    print()
    step2_build_vocab_and_tokenize_dataset(dataset_path)
    print()
    step3_token_to_id(dataset_path, lang1, lang2)
    print()
    step4_show_property(dataset_path, lang1, lang2)
    print()
    step5_gen_ref(dataset_path, lang1, lang2)


if __name__ == '__main__':
    main()
