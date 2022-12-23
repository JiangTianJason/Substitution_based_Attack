import sys
from opennre import encoder

try:
    sys.path.append('..')
    from server.adversary_utils import *
    from config import *
except ImportError:
    pass


logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=True,
                    choices=['tacred', 'wiki80'])
parser.add_argument('--model', '-m', required=True,
                    choices=['pcnn', 'cnn'])
parser.add_argument('--max_length', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')

args = parser.parse_args()

# Initialize model
rel2id = json.load(open(RELATION_PATH_DICT[args.dataset], 'r'))
id2rel = {v: k for k, v in rel2id.items()}
device = torch.device('cuda:0')

word2id = json.load(open('./pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('./pretrain/glove/glove.6B.50d_mat.npy')

# Define the sentence encoder
if args.encoder == 'pcnn':
    sentence_encoder = opennre.encoder.PCNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
elif args.encoder == 'cnn':
    sentence_encoder = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
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
model.load_state_dict(torch.load(MODEL_PATH_DICT[args.dataset][args.model],
                                 map_location=device)['state_dict'],False)
model.to(device)
model.eval()
model = REClassifier(model, rel2id, id2rel, device)

# Create directory for files containing filtered samples
target_dir = os.path.join(DATA_ROOT, args.dataset, args.model)
os.makedirs(target_dir, exist_ok=True)

for data_type in DATA_TYPES[args.dataset]:
    input_file = os.path.join(os.path.join(
        DATA_ROOT, args.dataset, data_type + '.txt'))
    output_file = os.path.join(target_dir, data_type + '_correct.index')

    # Filter out correct samples
    correct_samples = []
    samples = [json.loads(line) for line in open(input_file, 'r').readlines()]
    with open(output_file, 'w') as f:
        for idx, sample in enumerate(tqdm(samples)):
            label, _ = model.infer(sample)
            if label == sample['relation']:
                f.write(str(idx) + '\n')
