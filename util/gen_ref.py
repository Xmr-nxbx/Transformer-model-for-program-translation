import json
import os


def gen_ref_process(token_seq):
    text = ""
    for seq in token_seq:
        text += seq
    text = text.replace("$$", "").replace('▁', " ").strip()
    return text


def generate_reference(dataset_path, test_path, lang1, lang2):
    keyword = [lang1, lang2]
    refs = [[], []]
    lang1 = lang1 + "_raw"
    lang2 = lang2 + "_raw"
    with open(test_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for i in range(len(dataset)):
        refs[0].append(gen_ref_process(dataset[i][lang1]))  # list
        refs[1].append(gen_ref_process(dataset[i][lang2]))

    for i in range(len(keyword)):
        with open(os.path.join(dataset_path, keyword[i]), "w", encoding="utf-8") as f:
            f.write("\n".join(refs[i]))

