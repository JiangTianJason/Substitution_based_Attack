##########################################relation classification####################################


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
# # Training for wiki80 and tacred dataset
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', '-m', type=str, required=True,
#                     help='Full path for saving weights during training')
# parser.add_argument('--encoder_name', '-e', choices=['bert', 'bertentity'], default='bert',
#                     help=('Encoder model, choose from BERT (use [CLS] representation)'
#                           ' or BERT-entity (use concatenation of two entity representations)'))
# parser.add_argument('--restore', action='store_true',
#                     help='Whether to restore model weights from given model path')
# parser.add_argument('--train_path', '-t', type=str, required=True,
#                     help='Full path to file containing training data')
# parser.add_argument('--valid_path', '-v', type=str, required=True,
#                     help='Full path to file containing validation data')
# parser.add_argument('--relation_path', '-r', type=str, required=True,
#                     help='Full path to json file containing relation to index dict')
# parser.add_argument('--num_epochs', '-n', type=int, default=10,
#                     help='Number of training epochs')
# parser.add_argument('--max_seq_len', '-l', type=int, default=128,
#                     help='Maximum sequence length of bert model')
# parser.add_argument('--batch_size', '-b', type=int, default=64,
#                     help='Batch size for training and testing')
# parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
#                     help='Metric chosen for evaluation')
# parser.add_argument('--pretrain_path', '-p', type=str,
#                     help='Path to pretrained bert-base model weights')
#
# args = parser.parse_args()
#
# rel2id = json.load(open(args.relation_path))
#
# # Define the sentence encoder
# encoders = {'bert': encoder.BERTEncoder,
#             'bertentity': encoder.BERTEntityEncoder}
# sentence_encoder = encoders[args.encoder_name.lower()](
#     max_length=args.max_seq_len,
#     pretrain_path=args.pretrain_path,
#     mask_entity=False
# )
#
# # Define the model
# model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
#
# # Define the whole training framework
# framework = opennre.framework.SentenceRE(
#     train_path=args.train_path,
#     val_path=args.valid_path,
#     test_path=args.valid_path,
#     model=model,
#     ckpt=args.model_path,
#     batch_size=args.batch_size,
#     max_epoch=args.num_epochs,
#     lr=2e-5,
#     opt='adamw'
# )
#
# # Restore from old model weights
# if args.restore:
#     framework.load_state_dict(torch.load(args.model_path)['state_dict'])
#     logging.info('Restored model weights from {}'.format(args.model_path))
#
# # Train the model
# framework.train_model(args.metric)



##########################################adversarial training####################################

import torch
import json
import pickle
import opennre
from opennre import encoder, model, framework
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Training for wiki80 and tacred dataset
parser = argparse.ArgumentParser()
parser.add_argument('--encoder_name', '-e', choices=['bert', 'bertentity'], default='bert',
                    help=('Encoder model, choose from BERT (use [CLS] representation)'
                          ' or BERT-entity (use concatenation of two entity representations)'))
parser.add_argument('--restore', action='store_true',
                    help='Whether to restore model weights from given model path')
parser.add_argument('--valid_path', '-v', type=str, required=True,
                    help='Full path to file containing validation data')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--num_epochs', '-n', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')
parser.add_argument('--batch_size', '-b', type=int, default=64,
                    help='Batch size for training and testing')
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric chosen for evaluation')
parser.add_argument('--pretrain_path', '-p', type=str,
                    help='Path to pretrained bert-base model weights')

parser.add_argument('--adversarial_path', type=str,
                    help='Path to adversarial samples')
parser.add_argument('--data_type', type=str,
                    help='Type of adversarial samples')
parser.add_argument('--victim_model', type=str,
                    help='Model Name')
parser.add_argument('--dataset_type', type=str,
                    help='wiki80 or tacred')

args = parser.parse_args()

model_path = None

for num in [1000,2000,3000,4000,5000]:
    output_file = r"./dataset/{}/{}/final_{}_{}.txt".format(args.dataset_type, args.victim_model, args.data_type,num)
    target_sample_in_generate = [eval(line)["adversary_samples"] for line in open(args.adversarial_path, 'r',encoding='utf-8').readlines()]
    original_sample = [eval(line) for line in open(r"./dataset/{}/train.txt".format(args.dataset_type), 'r', encoding='utf-8').readlines()]
    final_train_data = target_sample_in_generate[:num] + original_sample
    for data in final_train_data:
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(str(data) + "\n")
            f.flush()
            f.close()

    model_path = r"./model/{}/{}_adversarial_NV_{}.pt".format(args.dataset_type,args.victim_model,num)
    logging.info("Directory???----> {}".format(output_file))
    logging.info("Save model???----> {}".format(model_path))

    rel2id = json.load(open(args.relation_path))

    # Define the sentence encoder
    encoders = {'bert': encoder.BERTEncoder,
                'bertentity': encoder.BERTEntityEncoder}
    sentence_encoder = encoders[args.encoder_name.lower()](
        max_length=args.max_seq_len,
        pretrain_path=args.pretrain_path,
        mask_entity=False
    )

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=output_file,
        val_path=args.valid_path,
        test_path=args.valid_path,
        model=model,
        ckpt=model_path,
        batch_size=args.batch_size,
        max_epoch=args.num_epochs,
        lr=2e-5,
        opt='adamw'
    )

    # Train the model
    framework.train_model(args.metric)
