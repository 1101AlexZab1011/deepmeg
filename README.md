# DeepMEG

DeepMEG is a torch-based library for EEG/MEG signals decoding. It provides a comprehensive set of tools for applying deep learning models for decoding EEG and MEG signals. The library is built on PyTorch, making it easy to use and integrate with other PyTorch-based projects. One of the main ideas embedded in the philosophy of this library is ease of use, even without advanced knowledge in the field of programming and deep learning.

## Features

* Easy-to-use deep learning models for decoding EEG/MEG signals
* Model weights interpretation in spatial and temporal domains
* Easy extensibility and complementarity of the code
* Easy integration with other PyTorch-based projects

## Installation

To install DeepMEG, simply run:

<pre><div class="bg-black mb-4 rounded-md"><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs">pip install git+https://github.com/1101AlexZab1011/deepmeg
</code></div></div></pre>

## Usage

Here is a simple example of how to use DeepMEG for EEG/MEG signals decoding:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class="">python</span></div></div></pre>

<pre><div class="bg-black mb-4 rounded-md"><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">import deepmeg
import mne
import torch
import torchmetrics
import matplotlib.pyplot as plt

# Load EEG/MEG signals
epochs = mne.read_epochs("epochs.fif")
# Do your preprocessing if you need (e.g. baseline, cropping, resampling, filtering and so on)

# Create dataset. Do not forget to use zscore to scale data (the initially MEG/EEG data amplitude is many orders of magnitude less than 1, which is obviously impossible to learn due to the vanishing of gradients)
dataset = EpochsDataset(epochs, savepath='../epochs_dataset_content', transform=deepmeg.preprocessing.transforms.zscore)

# Create train and test data
train, test = torch.utils.data.random_split(dataset, [.7, .3])

# initialize the model
model = deepmeg.models.interpretable.LFCNN(
    n_channels=info['nchan'], # ~ number of channels
    n_latent=8, # ~ number of latent factors
    n_times=len(epochs.times), # ~ number of samples in epoch after preprocessing
    pool_factor=10, # ~ take each 10th sample from spatially filtered components
    n_outputs=len(set(epochs.events[:, 2])) # ~ number of output classes (number of events in epochs)
)
model.compile(
    torch.optim.Adam,
    torch.nn.BCEWithLogitsLoss(),
    torchmetrics.functional.classification.binary_accuracy,
    callbacks=[
        deepmeg.training.callbacks.PrintingCallback(), # print ongoing training history
        deepmeg.training.callbacks.EarlyStopping(patience=15, monitor='loss_val', restore_best_weights=True), # perform early stopping with restoring best weights
        deepmeg.training.callbacks.L2Reg(
            [
                'unmixing_layer.weight', 'temp_conv.weight',
            ], lambdas=.01
        ) # l2 regularization for weights of spatial- and temporal filtering layers
    ]
)

# trin the model
history = model.fit(train, n_epochs=150, batch_size=200, val_batch_size=60)

train_result = model.evaluate(train)
result = model.evaluate(test)

for measure, value in train_result.items():
    print(f'train_{measure}: {value}')

for measure, value in result.items():
    print(f'{measure}: {value}')

interpreter = deepmeg.interpreters.LFCNNInterpreter(model, test, epochs.info)
fig = interpreter.plot_branch(0, ['input', 'response', 'pattern'])
plt.show()

</code></div></div></pre>

## Documentation

At the moment, the documentation for DeepMEG is only available in the source codeAt the moment, the documentation is only available in the source code

## License

DeepMEG is open-source software released under the MIT license.
