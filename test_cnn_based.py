##########################################relation classification####################################


# import numpy as np
# import torch
# import json
# import pickle
# import opennre
# from opennre import encoder, model, framework
# import argparse
# import logging
#
# logging.basicConfig(level=logging.INFO)
#
# # Silent unimportant log messages
# for logger_name in ['transformers.configuration_utils',
#                     'transformers.modeling_utils',
#                     'transformers.tokenization_utils_base']:
#     logging.getLogger(logger_name).setLevel(logging.WARNING)
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', '-m', type=str, required=True,
#                     help='Full path for saving weights during training')
# parser.add_argument('--encoder', '-e', choices=['pcnn', 'cnn'], default='pcnn')
# parser.add_argument('--test_path', '-t', type=str, required=True,
#                     help='Full path to file containing testing data')
# parser.add_argument('--relation_path', '-r', type=str, required=True,
#                     help='Full path to json file containing relation to index dict')
# parser.add_argument('--max_length', '-l', type=int, default=128,
#                     help='Maximum sequence length of bert model')
# parser.add_argument('--batch_size', '-b', type=int, default=64,
#                     help='Batch size for training and testing')
#
# args = parser.parse_args()
#
# rel2id = json.load(open(args.relation_path, 'r'))
#
# word2id = json.load(open('./pretrain/glove/glove.6B.50d_word2id.json'))
# word2vec = np.load('./pretrain/glove/glove.6B.50d_mat.npy')
#
# # Define the sentence encoder
# if args.encoder == 'pcnn':
#     sentence_encoder = opennre.encoder.PCNNEncoder(
#         token2id=word2id,
#         max_length=args.max_length,
#         word_size=50,
#         position_size=5,
#         hidden_size=230,
#         blank_padding=True,
#         kernel_size=3,
#         padding_size=1,
#         word2vec=word2vec,
#         dropout=0.5
#     )
# elif args.encoder == 'cnn':
#     sentence_encoder = opennre.encoder.CNNEncoder(
#         token2id=word2id,
#         max_length=args.max_length,
#         word_size=50,
#         position_size=5,
#         hidden_size=230,
#         blank_padding=True,
#         kernel_size=3,
#         padding_size=1,
#         word2vec=word2vec,
#         dropout=0.5
#     )
# else:
#     raise NotImplementedError
#
# # Define the model
# model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
# model.to(torch.device('cuda:0'))
#
# # Define the whole training framework
# framework = opennre.framework.SentenceRE(
#     train_path=args.test_path,
#     val_path=args.test_path,
#     test_path=args.test_path,
#     model=model,
#     ckpt=args.model_path,
#     batch_size=args.batch_size,
#     max_epoch=1,
#     lr=2e-5,
#     opt='adamw'
# )
#
# framework.load_state_dict(torch.load(args.model_path)['state_dict'])
# result = framework.eval_model(framework.test_loader)
#
# # Print the result
# print('Accuracy on test set: {}'.format(result['acc']))
# print('Micro Precision: {}'.format(result['micro_p']))
# print('Micro Recall: {}'.format(result['micro_r']))
# print('Micro F1: {}'.format(result['micro_f1']))





######################################adversarial training####################################


import numpy as np
import torch
import json
import pickle
import opennre
from opennre import encoder, model, framework
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--encoder_name', '-e', choices=['pcnn', 'cnn'], default='pcnn')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')
parser.add_argument('--batch_size', '-b', type=int, default=64,
                    help='Batch size for training and testing')


parser.add_argument('--victim_model', type=str,
                    help='Model Name')
parser.add_argument('--dataset_type', type=str,
                    help='wiki80 or tacred')

args = parser.parse_args()

output_file = r"./dataset/{}/{}/test_adversarial.txt".format(args.dataset_type, args.victim_model)
file_prefix = "test_" if args.dataset_type == "tacred" else "val_"
if not os.path.exists(output_file):
    for root, dirs, files in os.walk(r'./dataset/{}/{}/'.format(args.dataset_type, args.victim_model)):
        for file in files:
            if file.startswith(file_prefix) and file.endswith(".txt"):
                logging.info("Load data???----> {}".format(file))
                try:
                    final_test_data = [eval(line)["adversary_samples"][0] for line in open(os.path.join(root,file), 'r',encoding='utf-8').readlines()]
                except KeyError:
                    final_test_data = [eval(line)["adversary_samples"] for line in open(os.path.join(root,file), 'r',encoding='utf-8').readlines()]

                for data in final_test_data:
                    with open(output_file, "a", encoding='utf-8') as f:
                        f.write(str(data) + "\n")
                        f.flush()
                        f.close()

else:
    temp_output_file = r"./dataset/{}/{}/{}_NV.txt".format(args.dataset_type, args.victim_model,file_prefix.strip("_"))
    final_test_data = [eval(line)["adversary_samples"] for line in
                       open(temp_output_file, 'r', encoding='utf-8').readlines()]
    output_file = r"./dataset/{}/{}/onlytest_NV.txt".format(args.dataset_type, args.victim_model)

    #####################test_adversarial.txt###########
    for data in final_test_data:
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(str(data) + "\n")
            f.flush()
            f.close()

rel2id = json.load(open(args.relation_path, 'r'))

for num in [1000,2000,3000,4000,5000]:

    word2id = json.load(open('./pretrain/glove/glove.6B.50d_word2id.json'))
    word2vec = np.load('./pretrain/glove/glove.6B.50d_mat.npy')

    # Define the sentence encoder
    if args.encoder_name == 'pcnn':
        sentence_encoder = opennre.encoder.PCNNEncoder(
            token2id=word2id,
            max_length=args.max_seq_len,
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

    model_path = r"./model/{}/{}_adversarial_NV_{}.pt".format(args.dataset_type,args.victim_model,num)

    logging.info("Load model???----> {}".format(model_path))

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model.to(torch.device('cuda:0'))

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=output_file,
        val_path=output_file,
        test_path=output_file,
        model=model,
        ckpt=model_path,
        batch_size=args.batch_size,
        max_epoch=1,
        lr=2e-5,
        opt='adamw'
    )

    framework.load_state_dict(torch.load(model_path)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    # Print the result
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))
