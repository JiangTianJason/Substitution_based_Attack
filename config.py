MAX_SEQ_LEN = 128
PORT = 9012
DEVICE = 'gpu'  # Could be 'cpu'
PRETRAIN_PATH = 'Substitution_based_Attack/pretrain/bert-base-uncased/'
DATA_ROOT = 'Substitution_based_Attack/dataset'
DATA_TYPES = {
    'wiki80': ['train', 'val'],
    'tacred': ['train', 'dev', 'test']
}
MODEL_PATH_DICT = {
    'wiki80': {
        'bert': 'Substitution_based_Attack/model/wiki80/bert.pt',
        'bertentity': 'Substitution_based_Attack/model/wiki80/bertentity.pt'
    },
    'tacred': {
        'bert': 'Substitution_based_Attack/model/tacred/bert.pt',
        'bertentity': 'Substitution_based_Attack/model/tacred/bertentity.pt'
    }
}
RELATION_PATH_DICT = {
    'wiki80': 'Substitution_based_Attack/dataset/wiki80/rel2id.json',
    'tacred': 'Substitution_based_Attack/dataset/tacred/rel2id.json'
}
