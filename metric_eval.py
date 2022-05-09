import os

from evaluator import bleu
from evaluator.CodeBLEU import get_code_bleu_scores


def main(data_path="./data"):
    dataset_list = os.listdir(data_path)
    select = 0
    if len(dataset_list) == 0:
        exit(0)
    elif len(dataset_list) > 1:
        for i, each in enumerate(dataset_list):
            print("%d -> %s" % (i, each))
        select = input("select dataset by Index:\n")
        assert select.isdigit() and len(dataset_list) > int(select) >= 0
        select = int(select)
    data_path = os.path.join(data_path, dataset_list[select])

    file_list = os.listdir(data_path)
    keys = []
    refs = {}
    for each in file_list:
        file_name = os.path.join(data_path, each)
        if os.path.isfile(file_name) and not each.endswith("json"):
            keys.append(each)
            refs[each] = file_name
    keys.sort()

    model_path = os.path.join(data_path, "model")
    model_list = os.listdir(model_path)
    forward_trans = {}
    backward_trans = {}
    for each in model_list:
        model_output_path = os.path.join(model_path, r"%s/out" % each)
        if os.path.exists(model_output_path):
            if "Swap" in each:
                backward_trans[each] = model_output_path
            else:
                forward_trans[each] = model_output_path

    if len(forward_trans) != 0:
        show_result(keys, refs[keys[1]], forward_trans)
    if len(backward_trans) != 0:
        keys.reverse()
        show_result(keys, refs[keys[1]], backward_trans)


def show_result(keys: list, ref_path: str, outs: dict):
    title = "from key1(%s) to key2(%s)" % (keys[0], keys[1])
    lang = keys[1]
    if lang == "cs":
        lang = "c_sharp"

    bleu_result_dict = {}
    em_result_dict = {}
    codebleu_result_dict = {}

    with open(ref_path, "r+", encoding="utf-8") as f:
        ref = f.read().strip()
    refs = [x.strip() for x in ref.split('\n')]

    for key in outs.keys():
        with open(outs[key], "r+", encoding="utf-8") as f:
            pre = f.read().strip()
        pres = [x.strip() for x in pre.split('\n')]
        count = 0
        assert len(refs) == len(pres)
        for i in range(len(refs)):
            r = refs[i]
            p = pres[i]
            if r == p:
                count += 1
        em = round(count / len(refs) * 100, 2)

        bleu_result_dict[key] = round(bleu._bleu(ref_path, outs[key]), 2)
        em_result_dict[key] = em
        codebleu_result_dict[key] = round(get_code_bleu_scores([ref], pre, lang) * 100, 2)

    print("=" * 10 + title + "=" * 10)
    print("model\t\t\t\t\tbleu\t\tem\t\tcodebleu")
    for key in outs.keys():
        print(key + "\t\t" +
              str(bleu_result_dict.get(key)) + "\t\t" +
              str(em_result_dict.get(key)) + "\t\t" +
              str(codebleu_result_dict.get(key)))
        print()


if __name__ == '__main__':
    main()
