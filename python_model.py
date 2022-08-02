import torch
from torch import nn, optim

from python_preprocessor import prepare_data


class Code2Vec(nn.Module):
    def __init__(self, value_vocab, path_vocab, embedding_size, tag_vocab):
        super().__init__()
        self.tag_vocab = tag_vocab
        self.embedding_size = embedding_size
        self.value_vocab = value_vocab
        self.path_vocab = path_vocab
        self.value_vocab_matrix = nn.Embedding(len(value_vocab.keys()), embedding_size, device=device)
        self.path_vocab_matrix = nn.Embedding(len(path_vocab.keys()), embedding_size, device=device)
        self.tag_vocab_matrix = nn.Embedding(len(tag_vocab.keys()), embedding_size, device=device)
        self.fc = nn.Linear(3 * embedding_size, embedding_size, device=device)
        self.attention = nn.Linear(embedding_size, 1, device=device)
        self.rep_size = 10
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, input_function):
        cs = torch.zeros(self.rep_size, self.embedding_size).to(device)
        for i, context in enumerate(input_function):
            value1_embedding = self.value_vocab_matrix(torch.tensor(self.value_vocab[context[0]], device=device))
            path_embedding = self.path_vocab_matrix(torch.tensor(self.path_vocab[context[1]], device=device))
            value2_embedding = self.value_vocab_matrix(torch.tensor(self.value_vocab[context[2]], device=device))
            context = torch.cat((value1_embedding, path_embedding, value2_embedding))
            c_tilde = torch.tanh(self.fc(context))
            cs[i] = c_tilde

        attention_weights = torch.softmax(self.attention(cs), dim=0).to(device)

        v = torch.zeros(self.embedding_size, device=device)
        for i, context in enumerate(cs):
            v += (context * attention_weights[i])

        q = torch.zeros(len(self.tag_vocab), device=device)
        denominator = sum([torch.exp(v @ self.tag_vocab_matrix(torch.tensor(j, device=device)))
                           for j in range(len(self.tag_vocab))])
        for i in range(len(self.tag_vocab)):
            q[i] = torch.exp(v @ self.tag_vocab_matrix(torch.tensor(i, device=device))) / denominator

        return q


def train_loop(net, n_epochs, training_set):
    print(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for e in range(1, n_epochs + 1):
        loss = None
        for func, label in training_set.items():
            optimizer.zero_grad()
            pred = net(func).to(device)
            label_vector = torch.zeros((len(net.tag_vocab)), device=device)
            label_vector[net.tag_vocab[label]] = 1
            loss = criterion(pred, label_vector)
            loss.retain_grad()
            loss.backward()
            optimizer.step()
        print(f"EPOCH: {e}, loss: {loss}")


def index_to_tag(index, vocab):
    for tag, i in vocab.items():
        if i == index:
            return tag
    return None


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    training_data, validation_data, test_data, \
        value_vocabulary, path_vocabulary, tag_vocabulary = prepare_data("./data/python.json")

    model = Code2Vec(
        value_vocab=value_vocabulary,
        path_vocab=path_vocabulary,
        embedding_size=100,
        tag_vocab=tag_vocabulary
    )
    train_loop(model, n_epochs=10, training_set=training_data)

    prediction = index_to_tag(torch.argmax(model(list(test_data.items())[0][0]), dim=0), tag_vocabulary)
    print(prediction)
