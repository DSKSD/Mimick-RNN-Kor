import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Hangulpy as hg
import datetime
import pickle
flatten = lambda l: [item for sublist in l for item in sublist]
from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence


class MimickRNN(nn.Module):
    
    def __init__(self,vocab,word_embed,char_embed,char_hidden,mlp_hidden,use_cuda=False):
        super(MimickRNN,self).__init__()
        
        self.word_embed = nn.Embedding(len(vocab),word_embed)
        self.vocab = vocab
        
        char_vocab = ['<pad>','<other>','ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 
              'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 
              'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
              'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ',
              'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ',
              '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',
              'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G',
              'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',"{","}"
               '-','(',')','!','~','?','[',']',',','.','/','<','>','#','@','$','%','^','&','*','_',
               '+','-','=',':',';',"'",'"']
        
        self.char_hidden = char_hidden
        self.char2index = {v:i for i,v in enumerate(char_vocab)}
        self.char_embed = nn.Embedding(len(self.char2index), char_embed)
        self.mimick_rnn = nn.LSTM(char_embed,char_hidden,1,batch_first=True,bidirectional=True)
        self.mimick_linear = nn.Sequential(nn.Linear(char_hidden*2,mlp_hidden),
                                                           nn.Tanh(),
                                                           nn.Linear(mlp_hidden,word_embed))
        
        self.use_cuda = use_cuda
        
    def cuda(self, device=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        self.use_cuda = False
        return self._apply(lambda t: t.cpu())

        
    def init_word_embed(self,pretrained_vectors):
        self.word_embed.weight = nn.Parameter(torch.from_numpy(pretrained_vectors).float())
        self.word_embed.requires_grad = False # 고정
    
    def init_char_hidden(self,size):
        hidden = Variable(torch.zeros(2,size,self.char_hidden))
        context = Variable(torch.zeros(2,size,self.char_hidden))
        if self.use_cuda:
            hidden = hidden.cuda()
            context = hidden.cuda()
        return hidden, context
    
    def prepare_single_char(self,token):
        idxs=[]
        for s in token:
            if hg.is_hangul(s):
                # 음소 단위 분해
                try:
                    emso = list(hg.decompose(s))
                    if emso[-1]=='':
                        emso.pop()
                except:
                    emso = s
                idxs.extend(list(map(lambda w: self.char2index[w], emso)))
            else:
                candit=s
                if s.isalpha():
                    candit='<alpha>'
                try:
                    idxs.append(self.char2index[candit])
                except:
                    idxs.append(self.char2index['<other>']) # '' 가 OTHER같이
        tensor = torch.LongTensor(idxs)
        tensor = Variable(tensor)
        return tensor
    
    def prepare_char(self,seq,index=None):
        seq = list(map(lambda v: self.prepare_single_char(v), seq))
        if index:
            forsort = list(zip(seq,index))
            forsort = sorted(forsort,key = lambda s: s[0].size(0),reverse=True)
            seq,index = list(zip(*forsort))
            seq,index = list(seq),list(index)
        else:
            seq = sorted(seq,key = lambda s: s.size(0),reverse=True)
        length = [s.size(0) for s in seq]
        max_length = max(length)
        seq = [torch.cat([s,Variable(torch.LongTensor([self.char2index['<pad>']]*(max_length-s.size(0))))]).view(1,-1) for s in seq]
        seq = torch.cat(seq)
        if index:
            return seq, length, Variable(torch.LongTensor(index))
        else:
            return seq, length
        
    def train_mimick(self,step,batch_size=32,lr=0.0001):
        print("start training mimic-rnn with %d batch_size" % batch_size)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=lr)
        for step_index in range(step):
            try:
                offset = 0
                iter_index = list(range(len(self.vocab)//batch_size + 1))
                for i in iter_index:
                    voca = self.vocab[offset:offset+batch_size]
                    index = list(range(offset,offset+batch_size))
                    offset+=batch_size
                    
                    inputs, lengths, index = self.prepare_char(voca,index)
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        index = index.cuda()
                    self.zero_grad()
                    outputs = self.mimick(inputs,lengths)
                    targets = self.word_embed(index)
                    loss = F.mse_loss(outputs,targets)
                    loss.backward()
                    optimizer.step()
                    if i % 100==0:
                        print("[%d/%d] [%d/%d] mean_loss : %.7f" % (step_index,step,i,len(iter_index),loss.data[0]))
            except KeyboardInterrupt:
                print("Early Stop!")
                break
            
    def mimick(self,inputs,lengths):
        hidden = self.init_char_hidden(inputs.size(0))
        embedded = self.char_embed(inputs)
        packed = pack_padded_sequence(embedded,lengths,batch_first=True)
        outputs, (hidden,context) = self.mimick_rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = torch.cat([h for h in hidden], 1) # concat
        return self.mimick_linear(hidden)
    
    def get_most_word_embedding(self,word,num=5):
        matrix = self.word_embed.weight
        inputs,lengths = self.prepare_char([word])
        if self.use_cuda: inputs = inputs.cuda()
        embedding = self.mimick(inputs,lengths)
        similarities = matrix.matmul(embedding.transpose(0,1))
        similarities = similarities.transpose(0,1)
        norm = matrix.norm(dim=1)*embedding.norm()
        similarities = similarities/norm

        _ , i = similarities.topk(num)
        index = i.data.tolist()[0]
        similar_words = [self.vocab[i] for i in index]
        
        return similar_words
        