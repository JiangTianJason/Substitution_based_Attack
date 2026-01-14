
import json
import os
total_attack_data = []

generated_on = "pcnn"

for attack_method in ["test_tb","test_pw","test_hf","test_gen",f"noun+verb+adj+adv_tacred_{generated_on}_xiawei"]:
    single_attack_data = []
    with open(fr"../dataset/tacred/{generated_on}/{attack_method}.txt","r",encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            single_attack_data.append(json_data["index"])
        f.close()
    total_attack_data.append(single_attack_data)

common_index = sorted([set(total_attack_data[0]) & set(total_attack_data[1]) & set(total_attack_data[2]) & set(total_attack_data[3]) & set(total_attack_data[4])][0])

file_number = int(len(common_index) / 200) + 1

###standardize the name of head and tail entity
standard_h_t = []
with open(fr"../dataset/tacred/test.txt", "r", encoding="utf-8") as f:     #### wiki80\val.txt，tacred\test.txt
    for line in f:
        json_data = json.loads(line)
        standard_h_t.append({"h":json_data["h"]["name"],"t":json_data["t"]["name"]})
    f.close()

for attack_method in ["test_tb","test_pw","test_hf","test_gen",f"noun+verb+adj+adv_tacred_{generated_on}_xiawei"]:
    data_to_txt = []
    with open(fr"../dataset/tacred/{generated_on}/{attack_method}.txt","r",encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            if json_data["index"] in common_index:
                try:
                    json_data["adversary_samples"][0]["h"] = standard_h_t[json_data["index"]]["h"]
                    json_data["adversary_samples"][0]["t"] = standard_h_t[json_data["index"]]["t"]
                except:
                    json_data["adversary_samples"]["h"] = standard_h_t[json_data["index"]]["h"]
                    json_data["adversary_samples"]["t"] = standard_h_t[json_data["index"]]["t"]
                data_to_txt.append(json_data)
        f.close()
    for i in range(file_number):
        filename = rf'./wiki80-tacred-generated_on_{generated_on}/deepseek-tacred-v3/{attack_method}/{i}.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w+',encoding="utf-8") as f:
            for data in data_to_txt[i * 200:(i * 200 + 200)]:
                try:
                    f.write(json.dumps({"index":data["index"],"token":data["adversary_samples"][0]["token"],"h":data["adversary_samples"][0]["h"],"t":data["adversary_samples"][0]["t"]}) + "\n")
                except:
                    f.write(json.dumps({"index":data["index"],"token":data["adversary_samples"]["token"],"h":data["adversary_samples"]["h"],"t":data["adversary_samples"]["t"]}) + "\n")
            f.close()

with open(rf'./wiki80-tacred-generated_on_{generated_on}/deepseek-tacred-v3/common_index.txt',"w+") as f:
    f.writelines(str(common_index))
    f.close()