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
    attack_method = "test_gen"
    generated_on = "pcnn"

    ###standardize the name of head and tail entity
    standard_h_t = []
    standard_h_t_type = "test" if dataset_name == "tacred" else "val"
    with open(fr"../dataset/{dataset_name}/{standard_h_t_type}.txt", "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            standard_h_t.append({"h": json_data["h"]["name"], "t": json_data["t"]["name"]})
        f.close()

    with open(fr"../dataset/{dataset_name}/rel2id.json", "r", encoding="utf-8") as f:
        rel2id = json.load(f)

    with open(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-v3/common_index.txt", "r", encoding="utf-8") as f:
        common_index = ast.literal_eval(f.readline())
        f.close()

    file_number,deepseek_predict = 0,[]
    while os.path.exists(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-v3/{attack_method}/results_{file_number}.txt"):
        deepseek_predict += extract_relations(fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-v3/{attack_method}/results_{file_number}.txt")
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
                        write_for_deepseek_r1.append({"index":json_data["index"],"token":json_data["adversary_samples"][0]["token"],"h":standard_h_t[json_data["index"]]["h"],"t":standard_h_t[json_data["index"]]["t"]})
                    deepseek_predict_idx += 1
            except:     ###nvjr
                if json_data["index"] in common_index:
                    if rel2id[json_data["adversary_samples"]["relation"]] != deepseek_predict[deepseek_predict_idx]:
                        write_for_deepseek_r1.append(
                            {"index": json_data["index"], "token": json_data["adversary_samples"]["token"],
                             "h":standard_h_t[json_data["index"]]["h"], "t": standard_h_t[json_data["index"]]["t"]})
                    deepseek_predict_idx += 1
        f.close()

    print(attack_method,"ASR after transffering to V3：",len(write_for_deepseek_r1) / len(common_index))

    ###Attack successfully, sent to attack r1 and write files，write once
    file_number = int(len(write_for_deepseek_r1) / 50) + 1
    for i in range(file_number):
        filename = fr"./wiki80-tacred-generated_on_{generated_on}/deepseek-{dataset_name}-r1/{attack_method}/{i}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename,"w+", encoding="utf-8") as f2:
            for j in write_for_deepseek_r1[i * 50:(i * 50 + 50)]:
                f2.write(json.dumps(j) + "\n")
            f2.close()