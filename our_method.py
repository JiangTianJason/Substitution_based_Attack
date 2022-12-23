import sys
from opennre import encoder
import torch
from server.adversary_utils import *
from config import *
import json
from nltk.corpus import wordnet as wn
from pattern.en import *
from config import *
import nltk
import copy

logging.basicConfig(level=logging.INFO)

victim_model = "bertentity"
data_type = "tacred"
last_index = 0


#########adversarial path, put it in dataset, named train_NV.txt#############
# path = r"./tmp/Adversarial_samples/train_NV_{}_{}.txt".format(data_type, victim_model)


#########output path#############
path = r"./tmp/noun+verb+adj+adv_{}_{}.txt".format(data_type, victim_model)


if os.path.isfile(path):
    last_index = [eval(line) for line in open(path, 'r', encoding='utf-8').readlines()][-1]["index"]
else:
    txt_file = open(path, 'w')

logging.info(path)

class Load_data():
    def __init__(self):
        pass

    def load_data(self, data_type,last_index):

        if data_type == "wiki80":
            data_to_be_used = "val"
        else:
            data_to_be_used = "test"

        ##############adversarial_training############
        # data_to_be_used = "train"
        ########################################


        correct_indices = [int(line.strip()) for line in open(
            './dataset/{}/{}/{}_correct.index'.format(data_type, victim_model,data_to_be_used), 'r').readlines()]
        samples = [json.loads(line) for line in open(
            './dataset/{}/{}.txt'.format(data_type,data_to_be_used), 'r').readlines()]

        if last_index == 0:
            correct_samples = [samples[idx] for idx in correct_indices]
        else:
            next_index = correct_indices.index(last_index)
            logging.info('Restore progress from index ' + str(next_index))
            correct_samples = [samples[idx] for idx in correct_indices[next_index + 1:]]
            correct_indices = correct_indices[next_index + 1:]

        rel2id = json.load(open(
            './dataset/{}/rel2id.json'.format(data_type), 'r'))
        id2rel = {v: k for k, v in rel2id.items()}

        data_list = [(idx, data) for idx, data in zip(correct_indices, correct_samples)]

        return data_list, rel2id, id2rel


punctuation = "~!@#$%^&*()_+`{}|[]:\";-\\='<>?,./"
forbidden_VB = ["is","has","have","were","was","had","been","am","being","be","are","will","feel","look","smell","sound","taste","seem","appear","get","become","turn","grow","make","come","go","fall","run","remain","keep","stay","continue","stand","rest","lie","hold"]        #linking words
forbidden_ADJ = ["most", "more"]  # filter adjectives
forbidden_ADV = ["most", "more"]  # filter adverbs
total_sample, success = 0, 0
all_data, rel2id, id2rel = Load_data().load_data(data_type,last_index)

device = torch.device('cuda:0')

if "pcnn" not in victim_model:
    sentence_encoder = {'bert': encoder.BERTEncoder,
                        'bertentity': encoder.BERTEntityEncoder}[victim_model](
        max_length=128,
        pretrain_path=PRETRAIN_PATH,
        mask_entity=False
    )
else:
    word2id = json.load(open('./pretrain/glove/glove.6B.50d_word2id.json'))
    word2vec = np.load('./pretrain/glove/glove.6B.50d_mat.npy')

    if victim_model == 'pcnn':
        sentence_encoder = opennre.encoder.PCNNEncoder(
            token2id=word2id,
            max_length=128,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=0.5
        )
    elif victim_model == 'cnn':
        sentence_encoder = opennre.encoder.CNNEncoder(
            token2id=word2id,
            max_length=128,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=0.5
        )
    else:
        raise NotImplementedError

model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model.load_state_dict(torch.load(MODEL_PATH_DICT["{}".format(data_type)]["{}".format(victim_model)],
                                 map_location=device)['state_dict'], False)
model.to(device)
model.eval()
model = REClassifier(model, rel2id, id2rel, device)

for data in all_data:  # walk through all samples，data be like：(index,{"token":[],"relation":NA})

    total_sample += 1

    pos_tags = nltk.pos_tag(data[1]["token"])

    i, result, stage1_flag, noun_phrases, prediction_origin = 0, [], False, [], model.infer(data[1])[1]

    ###############################POS tags of token to be substituted, need change#########################
    for word, pos in pos_tags:
        if pos in ["NNP", "NNPS"] and word not in punctuation:
            noun_phrases.append(i)

        # elif ("NN" in pos or "VB" in pos or "JJ" in pos or "RB" in pos)  and word not in punctuation and word not in forbidden_VB and word not in forbidden_ADJ and lemma(word) not in forbidden_VB:

        # elif ("NN" in pos or "VB" in pos or "JJ" in pos) and word not in punctuation and word not in forbidden_VB and word not in forbidden_ADJ and lemma(word) not in forbidden_VB:
        # elif ("NN" in pos or "VB" in pos or "RB" in pos) and word not in punctuation and word not in forbidden_VB and word not in forbidden_ADJ and lemma(word) not in forbidden_VB:

        elif ("NN" in pos or "VB" in pos) and word not in punctuation and word not in forbidden_VB and lemma(word) not in forbidden_VB:         ###Nouns and Verbs

            result.append((word, pos, i))
        i += 1

    ###############################No need to change#########################

    if result:  # Tokens that satisfied
        h_start = data[1]['h']["pos"][0]
        h_end = data[1]['h']["pos"][1]
        t_start = data[1]['t']["pos"][0]
        t_end = data[1]['t']["pos"][1]

        for h_index in range(h_start, h_end):
            noun_phrases.append(h_index)
        for t_index in range(t_start, t_end):
            noun_phrases.append(t_index)  # index of noun phrases

        final_result = [k for k in result if k[2] not in noun_phrases]  # List of tokens that satisfied and not noun phrases

        for k in final_result:
            word = k[0]
            if "NN" in k[1]:
                pos = wn.NOUN
                number = "single" if k[1] in ["NN", "NNP"] else "plural"
                try:
                    aeiou_n = True if data[1]["token"][k[2] - 1] in ["an", "An"] else False  # starts with a vowel
                except IndexError:
                    aeiou_n = False
            elif "VB" in k[1]:
                pos = wn.VERB
                tense = "present" if k[1] in ["VB", "VBG", "VBP", "VBZ"] else "past"
                number = "single" if k[1] in ["VBP", "VBZ"] else "plural"

            elif "RB" in k[1]:
                pos = wn.ADV
                if k[1] == "RBR":
                    csn = "comparative"
                elif k[1] == "RBS":
                    csn = "superlative"
                else:
                    csn = "normal"

            elif "JJ" in k[1]:
                pos = wn.ADJ
                try:
                    aeiou = True if data[1]["token"][k[2] - 1] == "an" else False  # starts with a vowel
                except IndexError:
                    aeiou = False
                if k[1] == "JJR":
                    csn_ADJ = "comparative"
                elif k[1] == "JJS":
                    csn_ADJ = "superlative"
                else:
                    csn_ADJ = "normal"

            if "NN" in k[1] or "VB" in k[1] or "RB" in k[1]:
                word_property = wn.synsets(word, pos=pos)
                synonyms = [j for synset in word_property for j in synset.lemma_names() if j != word]

                ###############synonyms + hyponyms####################

                xiawei = [j.lemma_names() for synset in word_property for j in synset.hyponyms() if
                          j.lemma_names() != word]
                hyponyms = [j for i in xiawei for j in i if "_" not in j and "-" not in j]

                substitution_list = synonyms + hyponyms

                ##############################################


            elif "JJ" in k[1]:
                try:
                    substitution_adj = wn.synsets(word, pos=pos)[0].similar_tos()

                    all_adjectives = [str(qwe).split("\'")[1].split(".")[0] for qwe in substitution_adj]

                    if aeiou:
                        substitution_list = [words for words in all_adjectives if words[0] in "aeiou"]
                    else:
                        substitution_list = [words for words in all_adjectives if words[0] not in "aeiou"]

                    if substitution_list is None:
                        continue  # None starts with a vowel

                except IndexError:              #Not found
                    continue

            drop_list, temp_sample = [], copy.deepcopy(data)

            if substitution_list:  # Not empty
                for backup in substitution_list:  # No "_" in backup words
                    if "_" not in backup:
                        almost = backup

                        if pos == wn.NOUN:
                            if (aeiou_n and almost[0] in "aeiou") or (
                                    not aeiou_n and almost[0] not in "aeiou"):  # Satisfied
                                substitute = pluralize(almost) if number == "plural" else almost
                            else:
                                continue
                        elif pos == wn.VERB:
                            final_number = SG if number == "single" else PL
                            substitute = conjugate(verb=almost, tense=tense, number=final_number)

                        elif pos == wn.ADV:
                            if csn == "comparative":
                                substitute = comparative(almost).split(" ")[-1]
                            elif csn == "superlative":
                                substitute = superlative(almost).split(" ")[-1]
                            else:
                                substitute = almost

                        elif pos == wn.ADJ:
                            if csn_ADJ == "comparative":
                                substitute = comparative(almost).split(" ")[-1]
                            elif csn_ADJ == "superlative":
                                substitute = superlative(almost).split(" ")[-1]
                            else:
                                substitute = almost

                        try:

                            temp_sample[1]["token"][k[2]] = substitute

                            prediction_result = model.infer(temp_sample[1])
                            if prediction_result[0] != data[1]["relation"]:
                                with open(path, 'a') as f:
                                    f.write(json.dumps({'index': data[0], 'adversary_samples': temp_sample[1],
                                                        'predictions': prediction_result}) + '\n')

                                success += 1
                                stage1_flag = True
                                break
                            else:  # Substitution Not Success
                                drop_list.append((substitute, prediction_origin - prediction_result[1]))  # change of the confidence
                                temp_sample[1]["token"][k[2]] = k[0]  # restore token

                        except IndexError:
                            print("data is--->", data)
                            print("substitute is------->", substitute)

                if stage1_flag:
                    break
                else:
                    if drop_list:
                        data[1]["token"][k[2]] = sorted(drop_list, key=lambda tup: tup[1], reverse=True)[0][
                            0]  # changes the confidence of the victim model in judging correct relation significantly.
                        prediction_origin = model.infer(data[1])[1]
                    else:
                        prediction_origin = model.infer(data[1])[1]

    else:
        continue  # No tokens can be substituted

print("Success rate is: ", success / total_sample)
print("Total sample: ", total_sample)
print("Success sample: ", success)