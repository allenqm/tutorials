import torch
from torch import nn
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchtext
import torchvision
import numpy as np

torch.cuda.is_available()


## Create Data Set
def generate_timeseries(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = torch.rand(4, batch_size, 1)
    time = torch.linspace(0,1, n_steps)
    series = 0.5*torch.sin((time-offsets1) * (freq1*10+10)) # wave1
    series = series + 0.2*(torch.sin((time - offsets2) * (freq2 * 20 + 20))) # wave2
    series = series + 0.1 * (torch.rand(batch_size, n_steps) - 0.5) # noise
    return series.view(batch_size, n_steps, 1)


## Create train/test split
def create_datasplits(batch_size, n_steps, n_forecast_steps=1, split_percentages=[.70,.20,.10]):
    series = generate_timeseries(batch_size, n_steps + n_forecast_steps)
    train_end_i = np.floor(series.shape[0]*split_percentages[0]).astype(np.int16)
    valid_end_i = np.floor(series.shape[0]*sum(split_percentages[0:2])).astype(np.int16)
    res = {}
    res['X_train'], res['y_train'] = series[0:train_end_i, :n_steps], series[0:train_end_i, -n_forecast_steps:]
    res['X_valid'], res['y_valid'] = (series[train_end_i:valid_end_i, :n_steps], 
        series[train_end_i:valid_end_i, -n_forecast_steps:])
    res['X_test'], res['y_test'] = series[valid_end_i:, :n_steps], series[valid_end_i:, -n_forecast_steps:]
    return res



## init network and optimizer
class CNN(nn.Module):
    def __init__(self, output_length=1):
        super().__init__()
        # P = (F - 1)/2
        self.conv1 = nn.Conv1d(in_channels=1 , out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=10*50, out_features=output_length) #output_length)

    def forward(self, xb):
        xb = xb.view(50,1,50)
        xb = self.conv1(xb)
        xb = self.conv2(xb)
        xb = self.conv3(xb)
        xb = xb.view(50, 10*50)
        xb = self.fc1(xb)
        return xb

def test_cnn(batch_size=50): #tests that the forward pass can execute without error
    # can include: shape matches expectation, all gradients are zero
    # backwards pass: gradients are non-zero. loss is not zero
    # step: weights are not equal to prior step
    cnn= CNN() #init model
    xb = generate_timeseries(batch_size, 50) #generate a minibatch
    #xb = xb.view(10,1,50)
    output = cnn(xb) # forward pass through model. use modle.double() if converting from numpy
    return output


cnn= CNN()
xb = generate_timeseries(10, 50)
xb = xb.view(10,1,50)
output = cnn(xb)


## test dataset creation (everything up to and including splitting into folds)
N_FORECAST_STEPS = 10
res = create_datasplits(batch_size=10000, n_steps=50, n_forecast_steps=N_FORECAST_STEPS)
dataset_train = TensorDataset(res['X_train'], res['y_train'])
dataset_valid = TensorDataset(res['X_valid'], res['y_valid'])
dataset_test = TensorDataset(res['X_test'], res['y_test'])

dataloader_train = DataLoader(dataset_train, batch_size=50)
dataloader_valid = DataLoader(dataset_valid, batch_size=50)
dataloader_test = DataLoader(dataset_test, batch_size=50)



## training loop with eval
epochs = 2
lr = 0.1
momentum = 0.9
model = CNN(output_length=N_FORECAST_STEPS)
opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = mse_loss

for epoch in range(epochs):
    for xb, yb in dataloader_train:
        output = model(xb)
        loss = criterion(output, yb.squeeze(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            valid_loss = sum(criterion(model(xb), yb.squeeze(-1)) for xb, yb in dataloader_valid)
            print(epoch, valid_loss / len(dataloader_valid))
            # some evaluation
            # some monitoring
            # progress

## more evaluation
res={}
for i, (xb, yb) in enumerate(dataloader_test):
    if i <=2:
        res[i] = {'output': model(xb).data.numpy(),
                  'yb' : yb.numpy()}
df_eval = pd.concat([pd.Series(res[0]['output'].ravel()), pd.Series(res[0]['yb'].ravel())], axis=1)
df_eval.columns = ['output','yb']
df_eval.plot()

def plot_multi_step_prediction_vs_truth(batch_id, sample_id):
    df_eval = pd.concat([pd.DataFrame(res[batch_id]['output']).loc[sample_id,:], 
                         pd.DataFrame(res[batch_id]['yb'].squeeze(-1)).loc[sample_id,:].T], axis=1)
    df_eval.columns = ['output','yb']
    return df_eval.plot()

## unittests

def test_output_shape_matches_expectation(data, expected_shape):
    pass

def test_no_nans_in_data(data):
    pass

def test_values_in_array_are_in_range(array, range):
    pass

def test_gradients_are_updating(ModelClass):
    pass

## integration tests
def create_gold_standard(pipeline):
    pass


def compare_to_gold_standard(data, tolerances=[shape, nulls, numeric, categorical]):
    """
    compare shape
    compare nulls
    """
    pass

## tdd
def we_have_datasplits():
    pass

def we_can_train_a_model():
    pass