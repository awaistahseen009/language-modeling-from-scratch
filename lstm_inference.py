import torch
from torch import nn
import torch.nn.functional as F
import pickle
from minbpe.basic import BasicTokenizer
class LSTM(nn.Module):
    def __init__(self, n_emb, vocab_size, hidden_size, context_length, return_sequences=False):
        super().__init__()
        self.n_emb = n_emb
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.emb = nn.Embedding(vocab_size, n_emb)
        self.Wxh = nn.Parameter(torch.empty(n_emb, hidden_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wch = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wy = nn.Parameter(torch.empty(hidden_size, vocab_size))
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.xavier_uniform_(self.Wch)
        nn.init.xavier_uniform_(self.Wy)
        self.By = nn.Parameter(torch.zeros(vocab_size))
        self.Bh = nn.Parameter(torch.zeros(hidden_size))
        self.Bc = nn.Parameter(torch.zeros(hidden_size))


    def forward(self, idx):
        B, T = idx.shape
        hidden = torch.zeros(B, self.hidden_size, device=idx.device)
        cell_state = torch.zeros(B , self.hidden_size , device = idx.device) 
        embedded = self.emb(idx)  # Shape: [B, T, n_emb]
        outputs = []
        # looping over timestamp and all of this calculation can be considered as a single cell
        for t in range(T):
            x_t = embedded[:, t, :]  # Shape: [B, n_emb]
            prev_cell_s = (cell_state @ self.Wch ) + self.Bc
            hidden = (x_t @ self.Wxh + hidden @ self.Whh + self.Bh)  # Shape: [B, hidden_size]
            f_t = nn.functional.sigmoid(hidden)
            i_t = nn.functional.sigmoid(hidden)
            c_t  = (i_t * nn.functional.tanh(hidden)) + (f_t * prev_cell_s)
            hidden = nn.functional.tanh(c_t) * nn.functional.sigmoid(hidden)
            out = hidden @ self.Wy + self.By  # Shape: [B, vocab_size]

            if self.return_sequences:
                outputs.append(out.unsqueeze(1))  # Shape: [B, 1, vocab_size]
        
        if self.return_sequences:
            return outputs, (hidden, c_t)
        return out, (hidden, c_t)
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

vocab_size = 3072
n_emb = 768
hidden_size = 256
EPOCHS = 10000
context_length = 20
block_size = 20
batch_size = 128
num_layers = 2
return_sequences = False
model = LSTM(n_emb,vocab_size, hidden_size, context_length, return_sequences)
model.load_state_dict(torch.load("lstm_language_model_urdu.pth", weights_only=True))


print("Model Instantiated")
print("Loading the tokenizer...\n")
tokenizer = BasicTokenizer()
tokenizer.load(model_file = "./output/tokenizer/tokenizer.model")
print("Tokenizer Loaded..\n")
# generation = tokenizer.decode(model.generate(idx=torch.zeros((1 , 1), dtype = torch.long), max_new_tokens=100)[0].tolist())
generation = tokenizer.decode(model.generate(idx = torch.tensor(tokenizer.encode("ہم نے سنا ہے کہ"), dtype = torch.long).unsqueeze(0), max_new_tokens=500)[0].tolist())
with open("read_urdu.txt", "w", encoding = "utf-8") as f:
    f.write(generation)
print(generation)