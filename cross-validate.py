#########cross-validation###########
from opennre import encoder
from server.adversary_utils import *
import json
from pattern.en import *
from config import *

logging.basicConfig(level=logging.INFO)

victim_model = "bertentity"
generated_against_model = "pcnn"
data_type = "tacred"

for method_name in ["NV"]:

    method = method_name

    path = r"./cross_NV/{}_{}_{}_on_{}.txt".format(data_type, method, generated_against_model, victim_model)

    if os.path.isfile(path):
        pass
    else:
        txt_file = open(path, 'w')

    logging.info(path)

    if data_type == "wiki80":
        data_to_be_used = "val"
    else:
        data_to_be_used = "test"


    #################Load adversarial samples###############
    samples = [json.loads(line) for line in open(
        './dataset/{}/{}/{}_{}.txt'.format(data_type,generated_against_model,data_to_be_used,method), 'r').readlines()]
    rel2id = json.load(open(
        './dataset/{}/rel2id.json'.format(data_type), 'r'))
    id2rel = {v: k for k, v in rel2id.items()}


    success = 0

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
        else:
            raise NotImplementedError

    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model.load_state_dict(torch.load(MODEL_PATH_DICT["{}".format(data_type)]["{}".format(victim_model)],
                                     map_location=device)['state_dict'], False)
    model.to(device)
    model.eval()
    model = REClassifier(model, rel2id, id2rel, device)


    for temp_sample in samples:
        try:
            prediction_result = model.infer(temp_sample["adversary_samples"][0])
            if prediction_result[0] != temp_sample["adversary_samples"][0]["relation"]:
                success += 1
        except KeyError:
            prediction_result = model.infer(temp_sample["adversary_samples"])
            if prediction_result[0] != temp_sample["adversary_samples"]["relation"]:
                success += 1

    with open(path, 'a') as f:
        f.write(json.dumps({'success': success,'total':str(len(samples)),'ration':str(success/len(samples))}))