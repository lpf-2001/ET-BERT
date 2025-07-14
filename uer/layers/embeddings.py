import torch
import math
import pdb
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
import numpy as np

class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosEmbedding(nn.Module):
    """
    GPT embedding consists of two parts:
    word embedding and position embedding.
    """

    def __init__(self, args, vocab_size):
        super(WordPosEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )

        emb = word_emb + pos_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosSegEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__()
        # self.convert = nn.Linear(in_features=128,out_features=128)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        print("vocab_size:",vocab_size,"args.emb_size:",args.emb_size)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        

        
        
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)
    
    def convert(self,src):
        # print(src.shape)
        result1 = src.view(src.shape[0],-1,10).sum(2)
        result2 = src.view(src.shape[0],-1,20).sum(2)
        result3 = src.view(src.shape[0],-1,40).sum(2)
        return result1,result2,result3

    def forward(self, src, seg):
        # src = src.float()
        # pdb.set_trace()
        
        src1,src2,src3 = self.convert(src)
        

        src1 = torch.clamp(src1, min=0, max=60004)
        src2 = torch.clamp(src2, min=0, max=60004)
        src3 = torch.clamp(src3, min=0, max=60004)
        # print(src)
        src1 = src1.long()
        src2 = src2.long()
        src3 = src3.long()
        
        
        word_emb1 = self.word_embedding(src1)
        word_emb2 = self.word_embedding(src2)
        word_emb3 = self.word_embedding(src3)
      
        # print(word_emb.shape)
        pos_emb1 = self.position_embedding(
            torch.arange(0, word_emb1.size(1), device=word_emb1.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb1.size(0), 1)
        )
        pos_emb2 = self.position_embedding(
            torch.arange(0, word_emb2.size(1), device=word_emb2.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb2.size(0), 1)
        )
        pos_emb3 = self.position_embedding(
            torch.arange(0, word_emb3.size(1), device=word_emb3.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb3.size(0), 1)
        )
        seg1 = np.ones((word_emb1.shape[0],src1.shape[1]),dtype=np.int32)
        seg1 = torch.LongTensor(seg1).to('cuda')
        seg2 = np.ones((word_emb2.shape[0],src2.shape[1]),dtype=np.int32)
        seg2 = torch.LongTensor(seg2).to('cuda')
        seg3 = np.ones((word_emb3.shape[0],src3.shape[1]),dtype=np.int32)
        seg3 = torch.LongTensor(seg3).to('cuda')
        seg_emb1 = self.segment_embedding(seg1)
        seg_emb2 = self.segment_embedding(seg2)
        seg_emb3 = self.segment_embedding(seg3)

        emb1 = word_emb1 + pos_emb1 + seg_emb1
        emb2 = word_emb2 + pos_emb2 + seg_emb2
        emb3 = word_emb3 + pos_emb3 + seg_emb3
        
        if not self.remove_embedding_layernorm:
            # print(emb)
            emb1 = self.layer_norm(emb1)
            emb2 = self.layer_norm(emb2)
            emb3 = self.layer_norm(emb3)
        emb1 = self.dropout(emb1)
        emb2 = self.dropout(emb2)
        emb3 = self.dropout(emb3)
        return emb1,emb2,emb3


class WordSinusoidalposEmbedding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, args, vocab_size):
        super(WordSinusoidalposEmbedding, self).__init__()
        if args.emb_size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(args.emb_size))
        self.max_seq_length = args.max_seq_length
        pe = torch.zeros(self.max_seq_length, args.emb_size)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, args.emb_size, 2, dtype=torch.float)
                *- (math.log(10000.0) / args.emb_size)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, _):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        word_emb = self.word_embedding(src)
        emb = word_emb * math.sqrt(word_emb.size(-1))
        emb = emb + self.pe[: emb.size(1)].transpose(0, 1)
        emb = self.dropout(emb)
        return emb
