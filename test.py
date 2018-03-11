from model import MimickRNN
import pickle, os
import torch
import argparse


THIS_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='급식충' ,
                        help='테스트 할 한국어 단어')
    
    
    token = parser.parse_args().token
    config = pickle.load(open(THIS_PATH+"/models/mimick_03_11.config","rb"))
    vocab = pickle.load(open(THIS_PATH+config['vocab_path'],"rb"))
    model = MimickRNN(vocab,config['word_embed'],config['char_embed'],config['char_hidden'],config['mlp_hidden'])
    model.load_state_dict(torch.load(THIS_PATH+config['model_path']))
    
    try:
        model.vocab.index(token)
        print("%s is in voca" % token)
    except:
        print("%s is not in voca" % token)
        
    similar = model.get_most_word_embedding(token)
    print(token, " is similar with ",", ".join(similar))
    