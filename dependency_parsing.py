import logging

dataset_type,victim_model,prefix_name = "tacred","bert","noun+verb_tacred_bert_xiawei"
output_file = r"./dependency_analysis/{}_{}_{}.txt".format(dataset_type, victim_model, prefix_name)
output_each = r"./dependency_analysis/{}_{}_{}_each.txt".format(dataset_type, victim_model, prefix_name)
logging.info("句法结构分析结果输出：----> {}".format(output_file))

file_name = r"./dependency_analysis/tacred_pcnn_noun+verb_tacred_pcnn_xiawei_each.txt"
all_indices = [eval(line) for line in open(file_name, 'r',encoding='utf-8').readlines()]
root,head,tail, head_dep_relation, tail_dep_relation = 0, 0,0,{},{}
for i in all_indices:
    if i["root"]:
        root += 1
    if i["head"]:
        head += 1
    if i["tail"]:
        tail += 1
    if i["head_dep_relation"]:
        key = i["head_dep_relation"]
        try:
            head_dep_relation[key] += 1
        except KeyError:
            head_dep_relation[key] = 1
    if i["tail_dep_relation"]:
        key = i["tail_dep_relation"]
        try:
            tail_dep_relation[key] += 1
        except KeyError:
            tail_dep_relation[key] = 1


head_order=sorted(head_dep_relation.items(),key=lambda x:x[1],reverse=True)
tail_order=sorted(tail_dep_relation.items(),key=lambda x:x[1],reverse=True)

print("当前文件：",file_name)
print("root:",root)
print("head:",head)
print("tail:",tail)
print("root_percent:",root / len(all_indices))
print("head_percent:",head / len(all_indices))
print("tail_percent:",tail / len(all_indices))
print("head_order:",head_order)
print("tail_order",tail_order)
