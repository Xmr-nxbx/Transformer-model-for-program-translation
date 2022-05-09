import json
import os

keyword = ['train', 'valid', 'test']


def get_formatted_data(path):
    source_path = os.path.join(path, "source")
    data_path = os.path.join(path, "data")
    files = os.listdir(source_path)
    dataset = {k: [] for k in keyword}
    for f in files:
        for k in dataset.keys():
            if f.startswith(k):
                dataset[k].append(os.path.join(source_path, f))
    assert len(dataset[keyword[0]]) == 2
    lang1 = dataset[keyword[0]][0].split(".")[-1]
    lang2 = dataset[keyword[0]][1].split(".")[-1]

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for k in dataset.keys():
        print("Preparing the dataset - %s.json" % k)
        formatted_data = {}

        for each in dataset[k]:
            lang = each.split(".")[-1]
            with open(each, "r", encoding="utf-8") as f:
                current = f.readlines()
                current = [each.strip() for each in current]
            formatted_data[lang] = current

        current_dataset = [{
            "%s_raw" % lang1: formatted_data[lang1][i],
            "%s_raw" % lang2: formatted_data[lang2][i]
        } for i in range(len(formatted_data[lang1]))
        ]
        with open(os.path.join(data_path, "%s.json" % k), "w", encoding="utf-8") as f:
            json.dump(current_dataset, f)

    return lang1, lang2
