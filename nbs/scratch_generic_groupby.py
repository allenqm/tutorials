import torch

def generic_groupby(X, groupby_dim, func):
    unique_values = X.unique(dim=groupby_dim, return_counts=False)
    for u in uniques: # groupby_dim comes first
        selected = torch.nonzero(X[:,] == u).squeeze()
        selected = torch.index_select(X, 0, selected)
        selected = func(selected, dim=-1) #assuming -1 is the feature dimension

def generate_test_data():
    group_tensor = torch.randint()
    return None

# testing

generate_test_data()


samples = torch.Tensor([
                     [0.1, 0.1],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 2
                     [0.4, 0.4],    #-> group / class 2
                     [0.0, 0.0]     #-> group / class 0
              ])

labels = torch.LongTensor([1, 2, 2, 0])
labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))

unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
res = res / labels_count.float().unsqueeze(1)