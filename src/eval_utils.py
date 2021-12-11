def accuracy(true, pred):
    acc = 0.0
    for i in range(len(true)):
        if true[i] == pred[i]:
            acc += 1
    acc = acc / len(true)
    return acc


def binary_f1(true, pred, selected_class=True):
    f1 = None
    ## YOUR CODE STARTS HERE (~10-15 lines of code) ##
    ## YOUR CODE ENDS HERE ##
    # calculate binary f1 on class
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(true)):
        if(true[i] == selected_class and pred[i] == selected_class):
            true_pos += 1
        elif(true[i] != selected_class and pred[i] == selected_class):
            false_pos += 1
        elif(true[i] == selected_class and pred[i] != selected_class):
            false_neg += 1
    precision = true_pos / (true_pos + false_pos + 1)
    recall = true_pos / (true_pos + false_neg + 1)
    f1 = 2 * (precision * recall) / (precision + recall + 1)
    return f1


def binary_macro_f1(true, pred):
    averaged_macro_f1 = (binary_f1(true, pred, True) +
                         binary_f1(true, pred, False)) / 2

    return averaged_macro_f1
