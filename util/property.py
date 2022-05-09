import json


def show_dataset_property(training_set, lang1, lang2):
    keys = [lang1, lang2]
    lang1 = lang1 + "_raw"
    lang2 = lang2 + "_raw"
    # 0: [0: 60)  1: [60: 130)  2: [130: 210)  3: [210: 300)  4: [300: ]
    word_counter = [[0] * 5 for _ in range(2)]
    max_word = [0] * 2

    word_sum = [0] * 2
    with open(training_set, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_step = 0
    dataset_line_step = [0] * 2
    for datasets in dataset:
        dataset_step += 1

        def count_func(index, data):
            data = [[word for word in line if len(word) != 0] for line in data]
            data = [line for line in data if len(line) != 0]
            dataset_line_step[index] += len(data)

            word_len = sum([len(line) for line in data])
            word_sum[index] += word_len
            if word_len > max_word[index]:
                max_word[index] = word_len
            if word_len >= 0 and word_len < 60:
                word_counter[index][0] += 1
            elif word_len >= 60 and word_len < 130:
                word_counter[index][1] += 1
            elif word_len >= 130 and word_len < 210:
                word_counter[index][2] += 1
            elif word_len >= 210 and word_len < 300:
                word_counter[index][3] += 1
            else:
                word_counter[index][4] += 1

        count_func(0, datasets[lang1])
        count_func(1, datasets[lang2])

    for i in range(len(keys)):
        print('\nkeys: %s, statistics' % keys[i])
        print('word:')
        print(
            'max: %d  avg: %f  sum: %d  counter-all: %d  [0: 60): %d  [60: 130): %d  [130: 210): %d  [210: 300): %d  [300: ] : %d' % (
            max_word[i], word_sum[i] / dataset_step, word_sum[i], sum(word_counter[i]), *(word_counter[i])))
