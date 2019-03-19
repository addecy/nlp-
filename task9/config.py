import argparse

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=5,type=int)
    parser.add_argument('--attention_size',default=64,type=int)
    parser.add_argument('--hidden_layers',default=3,type=int)
    parser.add_argument('--hidden_units',default=64,type=int)
    return parser.parse_args()