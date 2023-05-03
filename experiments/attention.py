"""in the encoder the multiheaded attention takes in input the queries the values and the keys
then the input goes through to a normalization, a FFNN and another normalization
every sublayer block is surrounded by skip connection,the input that skipped
and the output of the sublayer are then added togheter into the Add & Norm
The output of the encoder is sent into the MultiHeaded of the decoder
for the values and the keys, while the queries are taken from the previous decoder
sublayer.
The first multiheaded attention of the decoder is MASKED and i didn't get why (the reason appears
to be the fact that we inject the target into the decoder, this can bring the decoder to learn a simple mapping
between the target and the desired output, we don't want this
ATTENTION MECHANISM
"""

import torch
import torch.nn as nn

"""here we create the class self attention: the class takes inherits form Module(the nn layer class of pytorch) and takes in input the embedding size(the size of the embedding of the input sequence) and the number of the heads
the class is composed by 3 linear layers: values, keys and queries, each of these layers applies a linear transformation to the input. The input size will be:
(batch_size, seq_len, heads, head_dim), the output of each layer will have the same size of the input. The last fc_out layer will still apply a linear transformation to the input, the input 
of this layer will have a size of (batch_size, seq_len*heads, head_dim) while the output will have a size of (batch_size, seq_len, embed_size).
the attention layer will take the input and reshape it to (batch_size, seq_len, heads, head_dim). Each linear layer will get in input #head tensors with size(batch_size, seq_len, head_dim), thes inputs will be projected respectively the queries, keys and values
 spaces. The queries and the keys will be multiplied togheter and the result will be normalized by the square root of the head_dim. The result will be multiplied by the values and the result will be the output of the attention layer.
 more schematically:
 1) the input tensor (batch_size, seq_len, embed_size) is reshaped to (batch_size, seq_len, heads, head_dim)
 2) the input is passed to the queries, keys and values linear layers, the output of each layer will have the same size of the input
 3) the queries and the keys are multiplied togheter and the result is normalized by the square root of the head_dim
 4) the results are softmaxed obtaining the attention weights, the values are multiplied by the attention weights for each head
 5) the results are concatenated obtaining a tensor of size (batch_size, seq_len, heads*head_dim=embed_size)
 6) the result is passed to the fc_out layer that multiplies it by a weight matrix obtaining in output a tensor of size (batch_size, seq_len, embed_size)"""
class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        #get the number of training examples
        N = query.shape[0] #how many samples we send in at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)#we reshape the last dimension (256) to 8x32 in order to split the embedding into 8 pieces
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)#we pass the values to the linear layer
        keys = self.keys(keys)#we pass the keys to the linear layer
        query = self.queries(query)#we pass the queries to the linear layer

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])#we use einsum to perform the dot product between the queries and the keys
        #queries shape: (N, query_len, heads, head_dim)
        #keys shape: (N, key_len, heads, head_dim)
        #energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))#we apply the mask to the energy tensor

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)#we apply the softmax to the energy tensor

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)#we apply the attention to the values
        #attention shape: (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, head_dim)
        #after einsum (N, query_len, heads, head_dim)
        # we concatenate the output of each head and project the input into a single output tensor of the same size of the input tensor with that reshape

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)#we create the attention layer
        self.norm1 = nn.LayerNorm(embed_size)#we create the first normalization layer
        self.norm2 = nn.LayerNorm(embed_size)#we create the second normalization layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)#we apply the attention layer
        x = self.dropout(self.norm1(attention + query))#we add the skip connection and apply the first normalization layer
        forward = self.feed_forward(x)#we apply the feed forward layer
        out = self.dropout(self.norm2(forward + x))#we add the skip connection and apply the second normalization layer
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size#we save the embedding size
        self.device = device#we save the device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)#we create the embedding layer
        self.position_embedding = nn.Embedding(max_length, embed_size)#we create the position embedding layer
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)])#we create the transformer blocks
        self.dropout = nn.Dropout(dropout)#we create the dropout layer

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)#we create the positions tensor
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))#we apply the dropout to the sum of the word embedding and the position embedding
        for layer in self.layers:#we apply the transformer blocks
            out = layer(out, out, out, mask) #the key and the query and the value are the same in the encoder
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)#we create the attention layer
        self.norm = nn.LayerNorm(embed_size)#we create the normalization layer
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)#we create the transformer block
        self.dropout = nn.Dropout(dropout)#we create the dropout layer

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)#we apply the attention layer
        query = self.dropout(self.norm(attention + x))#we add the skip connection and apply the normalization layer
        out = self.transformer_block(value, key, query, src_mask)#we apply the transformer block
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device#we save the device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)#we create the embedding layer
        self.position_embedding = nn.Embedding(max_length, embed_size)#we create the position embedding layer
        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])#we create the decoder blocks
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)#we create the output layer
        self.dropout = nn.Dropout(dropout)#we create the dropout layer

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)#we create the positions tensor
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))#we apply the dropout to the sum of the word embedding and the position embedding
        for layer in self.layers:#we apply the decoder blocks
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)#we apply the output layer
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)#we create the encoder
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)#we create the decoder
        self.src_pad_idx = src_pad_idx#we save the source padding index
        self.trg_pad_idx = trg_pad_idx#we save the target padding index
        self.device = device#we save the device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)#we create the source mask
        #src shape: (N, src_len)
        #src_mask shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len).to(self.device)#we create the target mask
        #trg_mask shape: (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)#we create the source mask
        trg_mask = self.make_trg_mask(trg)#we create the target mask
        enc_src = self.encoder(src, src_mask)#we apply the encoder
        out = self.decoder(trg, enc_src, src_mask, trg_mask)#we apply the decoder
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
