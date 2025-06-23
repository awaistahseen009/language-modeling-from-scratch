import torch
from torch import nn
import torch.nn.functional as F
import pickle
from minbpe.basic import BasicTokenizer
class RNN(nn.Module):
    def __init__(self, n_emb, vocab_size, hidden_size, context_length, return_sequences=False):
        super().__init__()
        self.n_emb = n_emb
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.emb = nn.Embedding(vocab_size, n_emb)
        self.Wxh = nn.Parameter(torch.empty(n_emb, hidden_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wy = nn.Parameter(torch.empty(hidden_size, vocab_size))
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.xavier_uniform_(self.Wy)
        self.By = nn.Parameter(torch.zeros(vocab_size))
        self.Bh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, idx):
        B, T = idx.shape
        hidden = torch.zeros(B, self.hidden_size, device=idx.device)
        embedded = self.emb(idx)  # Shape: [B, T, n_emb]
        outputs = []
        
        for t in range(T):
            x_t = embedded[:, t, :]  # Shape: [B, n_emb]
            hidden = torch.tanh(x_t @ self.Wxh + hidden @ self.Whh + self.Bh)   # Shape: [B, hidden_size]
            out = hidden @ self.Wy + self.By  # Shape: [B, vocab_size]
            if self.return_sequences:
                outputs.append(out.unsqueeze(1))  # Shape: [B, 1, vocab_size]
        
        if self.return_sequences:
            return outputs, hidden
        return out, hidden
    
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


# Setting the hyperparameters
vocab_size = 3072
n_emb = 768
hidden_size = 256
EPOCHS = 20000
context_length = 20
block_size = 20
batch_size = 64
num_layers = 2
return_sequences = False
model = RNN(n_emb,vocab_size, hidden_size, context_length,return_sequences = False)
print("Model Instantiated")

# Loading the tokenizer
print("Loading the tokenizer...\n")
tokenizer = BasicTokenizer()
tokenizer.load(model_file = "./output/tokenizer/tokenizer.model")
print("Tokenizer Loaded..\n")

data = pickle.load(open("data.pkl", "rb"))
print("Data Loaded Successfully..\n")
train_size = int(0.8* len(data))
test_size = int(0.1 * len(data))
train_data = torch.tensor(data[:train_size], dtype = torch.long)
val_data = torch.tensor(data[train_size: train_size + test_size], dtype = torch.long)
test_data = torch.tensor(data[-test_size:], dtype = torch.long)

print(f"data sizes: {len(train_data)}, {len(val_data)} ,{ len(test_data)}")

def return_batch(split:str):
    data = train_data if split == "train" else val_data
    idx = torch.randint(1 , len(data) - block_size, (batch_size,))
    xs = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return xs, y

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
loss_fn = nn.CrossEntropyLoss()
print("Model Training Starte\n")
for i in range(EPOCHS):
    x, y = return_batch("train")
    logits, h = model(x)
    # print(logits.shape, y.shape)
    loss =  loss_fn(logits, y[:, -1])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if  i %1000==0:
        model.eval()
        x_val , y_val = return_batch("val")
        logits_val , h_val = model(x_val)
        val_loss =  loss_fn(logits_val, y_val[:, -1])
        print(f"After Epoch: {i+1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
        model.train()

print(loss.item())
print("Model Trained Successfully...\n")
print("Saving the model..\n")
torch.save(model.state_dict(), 'rnn_language_model_urdu.pth')
print("Model saved successfully..\n")