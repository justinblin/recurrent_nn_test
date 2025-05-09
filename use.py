import torch
import preprocess
import postprocess
from rnn import CharRNN
import glob
import os

def use(rnn:CharRNN, name:str, labels_unique:list[str]) -> tuple[str, int]:
    rnn.eval()
    with torch.no_grad():
        name_tensors = preprocess.string_to_tensor(name)
        # print(name_tensors)
        output_tensor = rnn(name_tensors)
        guess, guess_index = postprocess.label_from_output(output_tensor, labels_unique)
        return guess, guess_index

# get the list of unique languages
labels_unique = []
text_files = glob.glob(os.path.join('./data/names', '*.txt'))
for filename in text_files: # go thru each of the text files, get the name of the file, add to labels set
    labels_unique.append(os.path.splitext(os.path.basename(filename))[0])

# load the trained model
rnn = torch.load('./my_model', weights_only = False)

# use the model
name_to_guess = 'Hwang'
guess, guess_index = use(rnn, name_to_guess, labels_unique)
print('My RNN guessed ' + name_to_guess + ' is ' + guess + ', index: ' + str(guess_index)) # print the name that's guessed