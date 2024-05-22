import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_size = int(0.8 * len(tokenized_datasets))
val_size = len(tokenized_datasets) - train_size
train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(output[:, -1, :])

VOCAB_SIZE = len(tokenizer)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = VOCAB_SIZE
N_LAYERS = 2

torch.manual_seed(42)
model1 = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(device)

torch.manual_seed(24)
model2 = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(device)

def train_model(model, train_dataloader, val_dataloader, epochs=1, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            print("Batch: ", i + 1, end="\r")
            inputs = batch['input_ids'].squeeze(1).to(device)
            targets = batch['input_ids'].squeeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['input_ids'].squeeze(1).to(device)
                targets = batch['input_ids'].squeeze(1).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets[:, 0])
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        model.train()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# train_losses1, val_losses1 = train_model(model1, train_dataloader, val_dataloader,epochs=20)
# train_losses2, val_losses2 = train_model(model2, train_dataloader, val_dataloader,epochs=20)

# def plot_losses(train_losses, val_losses, title, filename):
#     epochs = range(1, len(train_losses) + 1)
#     plt.figure()
#     plt.plot(epochs, train_losses, 'bo-', label='Training loss')
#     plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
#     plt.title(title)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(filename)
#     plt.close()

# plot_losses(train_losses1, val_losses1, 'Model 1 Training and Validation Loss', 'model1_loss.png')
# plot_losses(train_losses2, val_losses2, 'Model 2 Training and Validation Loss', 'model2_loss.png')

# torch.save(model1.state_dict(), 'model1.pth')
# torch.save(model2.state_dict(), 'model2.pth')


model1.load_state_dict(torch.load('RNN/model1.pth'))
model2.load_state_dict(torch.load('RNN/model2.pth'))

# Ensure models are in evaluation mode
model1.eval()
model2.eval()

# Extract embeddings
model1_embeddings = model1.embedding.weight.detach().cpu().numpy()
model2_embeddings = model2.embedding.weight.detach().cpu().numpy()

# Compare embeddings using cosine similarity
def compare_embeddings(embeddings1, embeddings2):
    similarities = cosine_similarity(embeddings1, embeddings2)
    mean_similarity = np.mean(similarities)
    return similarities, mean_similarity

embedding_similarities, mean_similarity = compare_embeddings(model1_embeddings, model2_embeddings)

print("Embedding Similarities (Cosine Similarity Matrix):")
print(embedding_similarities)
print(f"Mean Cosine Similarity: {mean_similarity:.4f}")


# Visualize embeddings with t-SNE
def visualize_embeddings(embeddings1, embeddings2):
    tsne = TSNE(n_components=2, random_state=42)
    combined_embeddings = np.vstack((embeddings1, embeddings2))
    tsne_results = tsne.fit_transform(combined_embeddings)

    plt.figure(figsize=(12, 6))
    plt.scatter(tsne_results[:VOCAB_SIZE, 0], tsne_results[:VOCAB_SIZE, 1], c='blue', label='Model 1')
    plt.scatter(tsne_results[VOCAB_SIZE:, 0], tsne_results[VOCAB_SIZE:, 1], c='red', label='Model 2')
    plt.legend()
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('embedding_visualization.png')
    plt.close()

visualize_embeddings(model1_embeddings, model2_embeddings)
