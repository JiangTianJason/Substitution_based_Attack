import os
import ast
import json

def extract_relations(file_path):
    all_relations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                all_relations.append(json.loads(line)["predicted_relation_id"])
            except:
                pass

    return all_relations


if __name__ == "__main__":
    dataset_name = "tacred"
    attack_method = "test_tb"
    generated_on = "pcnn"

    with open(fr"../dataset/{dataset_name}/rel2id.json", "r", encoding="utf-8") as f:
        rel2id = json.load(f)

    r1_number,common_index = 0,[]
    while os.path.exists(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-r1/{attack_method}/{r1_number}.txt"):
        with open(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-r1/{attack_method}/{r1_number}.txt", "r", encoding="utf-8") as f:
            for line in f:
                common_index.append(json.loads(line)["index"])
            f.close()
        r1_number += 1

    file_number,deepseek_predict = 0,[]
    while os.path.exists(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-r1/{attack_method}/results_{file_number}.txt"):
        deepseek_predict += extract_relations(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-r1/{attack_method}/results_{file_number}.txt")
        file_number += 1

    assert len(common_index) == len(deepseek_predict)

    ###Compare the correct labels and the output of deepseek
    write_for_deepseek_r1,deepseek_predict_idx = [],0
    with open(fr"../dataset/{dataset_name}/{generated_on}/{attack_method}.txt", "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            try:
                if json_data["index"] in common_index:
                    if  rel2id[json_data["adversary_samples"][0]["relation"]] != deepseek_predict[deepseek_predict_idx]:
                        write_for_deepseek_r1.append(1)     ###if attack successfully, append anything
                    deepseek_predict_idx += 1
            except:     ###nvjr
                if json_data["index"] in common_index:
                    if rel2id[json_data["adversary_samples"]["relation"]] != deepseek_predict[deepseek_predict_idx]:
                        write_for_deepseek_r1.append(1)
                    deepseek_predict_idx += 1
        f.close()


    with open(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-v3/common_index.txt", "r", encoding="utf-8") as f:
        bert_success_common = ast.literal_eval(f.readline())
        f.close()

    print(attack_method,"ASR after transffering to R1：",len(write_for_deepseek_r1) / len(bert_success_common))