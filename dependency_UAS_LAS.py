import logging
import stanza
from tqdm import tqdm

dataset_type,victim_model,prefix_name = "tacred","bert","noun+verb_tacred_bert_xiawei"    ############只喂我们的NV
output_file = r"./syntactic/{}_{}_{}.txt".format(dataset_type, victim_model, prefix_name)
logging.info("句法结构分析结果输出：----> {}".format(output_file))

def do_it(dataset_type,victim_model,prefix_name):
    all_indices = [eval(line) for line in open(r"./dataset/{}/{}/{}.txt".format(dataset_type,victim_model,prefix_name), 'r',encoding='utf-8').readlines()]

    if dataset_type == "tacred":
        original_sample = [eval(line) for line in open(r"./dataset/{}/test.txt".format(dataset_type), 'r').readlines()]
    else:
        original_sample = [eval(line) for line in open(r"./dataset/{}/val.txt".format(dataset_type), 'r').readlines()]

    original_list = [original_sample[sample["index"]] for sample in all_indices]

    return original_list,all_indices


nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English


original_list,generated_list = do_it(dataset_type,victim_model,prefix_name)
total_UAS,total_LAS,total_sample = 0,0,len(original_list)
for text1,text2 in tqdm(zip(original_list,generated_list)):
    direction,both,total = 0,0,0

    text1 = " ".join(text1["token"])
    text2 = " ".join(text2["adversary_samples"]["token"])
    doc_origin = nlp(text1)
    doc_NV = nlp(text2)

    print(doc_origin.sentences[0].constituency)
    print(doc_NV.sentences[0].constituency)
    print("\n")


    #####Dependency##########
    result_origin = doc_origin.sentences[0].print_dependencies()
    print("\n")
    result_NV = doc_NV.sentences[0].print_dependencies()

    for i,j in zip(result_origin,result_NV):
        total += 1
        if i[1] == j[1]:
            direction += 1
            if i[2] == j[2]:
                both += 1


    #####UAS + LAS#########Dependency########

    total_UAS += direction / total
    total_LAS += both / total

print("UAS_avg:",total_UAS/total_sample)
print("LAS_avg:",total_LAS/total_sample)


f = open(output_file, "a", encoding='utf-8')
f.write("UAS_avg:" + str(total_UAS/total_sample) + "\n")
f.write("LAS_avg:" + str(total_LAS/total_sample) + "\n")
f.close()







################draw_nationality_distribution##################

# # import matplotlib.pyplot as plt
# # import numpy
# # import numpy as np
# #
# # y = np.array([1,1, 1, 2, 2,2,2,1])
# # def absolute_value(val):
# #     a  = numpy.round(val/100.*y.sum(), 0)
# #     return int(a)
# # plt.pie(y,
# #         labels=['Ethiopia','Uzbekistan','Kazakhstan','Yemen','Bangladesh','China','Pakistan','Ukraine'],
# #         explode=(0.01,0.01, 0.01, 0.01, 0.01,0.01,0.01,0.01),
# #         colors= ["burlywood","burlywood","burlywood","burlywood","burlywood","burlywood","burlywood","burlywood"],
# #         autopct=absolute_value)
# #
# # plt.savefig("nationality.jpeg" ,dpi=600, bbox_inches='tight')
# # plt.show()