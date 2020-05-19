import torch

def generic_groupby(X, groupby_dim, func):
    uniques = get_unique_values(X, groupby_dim)
    for u in uniques: # groupby_dim comes first
        selected = torch.nonzero(X[:,] == u).squeeze()
        selected = torch.index_select(X, 0, selected)
        selected = func(selected, dim=-1) #assuming -1 is the feature dimension

def generate_test_data():
    group_tensor = torch.randint()
    return None

# testing

generate_test_data()