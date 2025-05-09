# turn output tensor into a human-readable label
def label_from_output(output, output_labels):
    top_number, top_index = output.topk(1) # get list of 1 largest element and index
    label_index = top_index[0].item() # pull out index and convert from tensor into standard python
    return output_labels[label_index], label_index