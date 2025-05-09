# SET UP IMPORTS, GPU, DATASET
import torch
import os
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import preprocess
import postprocess
from dataset import NamesDataset
from rnn import CharRNN

# use GPU if possible
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.set_default_device(device)

# set up dataset
all_data = NamesDataset("data/names")

# split dataset into training and testing set
train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.8, .2, .0], 
                                                    generator = torch.Generator(device = device).manual_seed(123))


# CREATE/TRAIN NEURAL NETWORK
rnn = CharRNN(len(preprocess.allowed_char), 128, len(all_data.labels_unique))

# train neural network
def train(rnn, training_data, num_epoch:int = 10, batch_size:int = 64, report_every:int = 1, learning_rate:float = 0.2, criterion = nn.NLLLoss()):
    # track loss over time
    current_loss = 0
    all_losses = []
    rnn.train() # flag that you're starting to train now
    optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate) # stochastic gradient descent

    print(f'Start training on {len(training_data)} names')

    # go thru each epoch
    for epoch_index in range(num_epoch):
        # split training data into batches
        batches = list(range(len(training_data))) # make list of indices (for each tensor) going from 0 to length of training data
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // batch_size) # integer division AKA split list into batches of indices

        # go thru each batch
        for batch in batches:
            batch_loss = 0 # total loss for this batch
            # go thru each tensor in this batch
            for curr_elem in batch:
                # run forward and figure out the loss
                (label_tensor, name_tensor, label, name) = training_data[curr_elem]
                output = rnn.forward(name_tensor) # tensor that's outputted
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # run back propogation
            batch_loss.backward() # find out how much to change each weight/bias
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step() # apply the changes to weight/biases
            optimizer.zero_grad() # prevent exploding gradients

            current_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

        # log the current loss
        all_losses.append(current_loss / len(batches))
        if epoch_index % report_every == 0:
            print(f'Epoch {epoch_index}: average batch loss = {all_losses[-1]}')

        current_loss = 0 # reset loss so it doesn't build up in the tracking

    return all_losses

all_losses = train(rnn, train_set, num_epoch = 30, report_every = 2)

torch.save(rnn, "./my_model")

# show training results
plt.figure()
plt.plot(all_losses)
plt.show()

# TEST NEURAL NETWORK
def test(rnn, testing_data, classes):
    confusion_matrix = torch.zeros(len(classes), len(classes))
    percent_correct = 0

    rnn.eval() # turn on eval flag
    with torch.no_grad(): # don't record gradients
        for index in range(len(testing_data)): # go thru each test example
            (label_tensor, data_tensor, label, data) = testing_data[index]
            label_index = classes.index(label) # the index of the correct answer for this testcase
            output = rnn(data_tensor)
            guess, guess_index = postprocess.label_from_output(output, classes) # the guess and index of the guess

            if index % (len(testing_data)//10) == 0: # print 10 outputs
                print("guess: "+str(guess)+", guess_idx: "+str(guess_index)+", label: "+str(label)+", label_idx: "+str(label_index))

            confusion_matrix[guess_index][label_index] += 1
            if guess_index == label_index:
                percent_correct += 1

    # turn the total count into percentage (aka normalize)
    for row_index in range(len(confusion_matrix)):
        row_total = confusion_matrix[row_index].sum()
        if (row_total > 0):
            confusion_matrix[row_index] /= row_total
    percent_correct = (percent_correct*100)//len(testing_data)

    print(torch.round(confusion_matrix, decimals = 4))
    print(str(percent_correct) + "% correct")

    # plot the confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix.cpu().numpy()) # need to convert from gpu to cpu mem bcs numpy uses cpu
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel("Answer")
    plt.ylabel("Guess")
    plt.show()

test(rnn, test_set, all_data.labels_unique)

