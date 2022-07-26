import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from python_preprocessor import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Code2Vec(nn.Module):
    def __init__(self, value_vocab, path_vocab, embedding_size, tag_vocab):
        super().__init__()
        self.tag_vocab = tag_vocab
        self.embedding_size = embedding_size
        self.value_vocab = value_vocab
        self.path_vocab = path_vocab
        self.value_vocab_matrix = nn.Embedding(len(value_vocab.keys()), embedding_size)
        self.path_vocab_matrix = nn.Embedding(len(path_vocab.keys()), embedding_size)
        self.tag_vocab_matrix = nn.Embedding(len(tag_vocab.keys()), embedding_size)
        self.fc = nn.Linear(3 * embedding_size, embedding_size)
        self.attention = nn.Linear(embedding_size, 1)
        self.rep_size = 20

    def forward(self, input_function):
        cs = []
        for context in input_function[:self.rep_size]:
            value1_embedding = self.value_vocab_matrix(torch.tensor(self.value_vocab[context[0]]))
            path_embedding = self.path_vocab_matrix(torch.tensor(self.path_vocab[context[1]]))
            value2_embedding = self.value_vocab_matrix(torch.tensor(self.value_vocab[context[0]]))
            context = torch.cat((value1_embedding, path_embedding, value2_embedding))
            c_tilde = torch.tanh(self.fc(context))
            cs.append(c_tilde.detach().numpy())
        cs = np.array(cs)
        attention_weights = torch.softmax(self.attention(torch.tensor(cs)), dim=0)

        v = torch.zeros(self.embedding_size).to(device)
        for i, context in enumerate(cs):
            v += (context * float(attention_weights[i]))

        a = []
        for i in range(len(self.tag_vocab)):
            a.append(self.tag_vocab_matrix(torch.tensor(i)))
        q = []
        for i in range(len(self.tag_vocab)):
            enumerator = torch.exp(v @ a[i])
            denominator = sum([torch.exp(v @ a[j]) for j in range(len(self.tag_vocab))])
            q.append(enumerator / denominator)

        return torch.tensor(q)


def train_loop(model, n_epochs, training_data):
    print(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (ADAM is a fancy version of SGD)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for e in range(1, n_epochs + 1):
        loss = None
        for func, label in training_data.items():
            _label = label
            optimizer.zero_grad()
            prediction = model.forward(func)
            label = torch.tensor(model.tag_vocab[label]).to(device)
            loss = criterion(prediction, label)
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
        print(f"EPOCH: {e}, loss: {loss}")


if __name__ == '__main__':
    training_data, value_vocabulary, path_vocabulary, tag_vocabulary = prepare_data("./data/python.json")

    model = Code2Vec(
        value_vocab=value_vocabulary,
        path_vocab=path_vocabulary,
        embedding_size=100,
        tag_vocab=tag_vocabulary
    )
    train_loop(model, n_epochs=10, training_data=training_data)
