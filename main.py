import argparse
import network
import data
import image
import compress
import warnings
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# filter all runtime warnings
warnings.filterwarnings('ignore')

def main(hyperparameters):
    # convert hyperparameters into tuples
    hyperparameters = list(vars(hyperparameters).values())
    hyperparameters = tuple(hyperparameters)

    # training dataset for logistic regression class 0 and class 6
    logistic_train_1 = data.np.where((data.load_data(True)[1] == 0) | (data.load_data(True)[1] == 6))
    logistic_train_1 = data.load_data(True)[0][logistic_train_1[0]], data.load_data(True)[1][logistic_train_1[0]]
    binary_6_to_1 = data.np.where(logistic_train_1[1] == 6, 1, logistic_train_1[1])
    # compress via SVD
    compressed_train_1 = []
    for img in (logistic_train_1[0]):
        compressed_train_1.append(compress.compress_image(img.reshape(28,-1), hyperparameters[5]).reshape(28*28))
    compressed_train_1 = np.array(compressed_train_1)
    # original set
    logistic_train_1 = logistic_train_1[0], binary_6_to_1
    logistic_train_1 = data.append_bias(hyperparameters[3](logistic_train_1[0])[0]), logistic_train_1[1]
    logistic_train_1 = data.shuffle(logistic_train_1)
    # compressed set
    logistic_train_1_compressed = compressed_train_1, binary_6_to_1
    logistic_train_1_compressed = data.append_bias(
        hyperparameters[3](logistic_train_1_compressed[0])[0]), logistic_train_1_compressed[1]
    logistic_train_1_compressed = data.shuffle(logistic_train_1_compressed) 

    # testing dataset for logistic regression class 0 and class 6
    logistic_test_1 = data.np.where((data.load_data(False)[1] == 0) | (data.load_data(False)[1] == 6))
    logistic_test_1 = data.load_data(False)[0][logistic_test_1[0]], data.load_data(False)[1][logistic_test_1[0]]
    binary_6_to_1_2 = data.np.where(logistic_test_1[1] == 6, 1, logistic_test_1[1])
    # compress via SVD
    compressed_test_1 = []
    for img in (logistic_test_1[0]):
        compressed_test_1.append(compress.compress_image(img.reshape(28,-1), hyperparameters[5]).reshape(28*28))
    compressed_test_1 = np.array(compressed_test_1)
    # original set
    logistic_test_1 = logistic_test_1[0], binary_6_to_1_2
    logistic_test_1 = data.append_bias(hyperparameters[3](logistic_test_1[0])[0]), logistic_test_1[1]
    # compressed set
    logistic_test_1_compressed = compressed_test_1, binary_6_to_1_2
    logistic_test_1_compressed = data.append_bias(
        hyperparameters[3](logistic_test_1_compressed[0])[0]), logistic_test_1_compressed[1]
    logistic_test_1_compressed = data.shuffle(logistic_test_1_compressed)


    # # save sample images of each class
    # sample_image = []
    # original_train_data = data.load_data(True)
    # for i in range(10):
    #     train_data_ind = data.np.where(original_train_data[1] == i)
    #     # find the first image of its class
    #     selected = original_train_data[0][train_data_ind[0][0]]
    #     compressed_selected = compress.compress_image(selected.reshape(28, -1), hyperparameters[5])
    #     image.export_image(selected, name="out_images/" + 'class' + str(i) + '.png')
    #     image.export_image(compressed_selected, name="out_images/" + 'class' + str(i) + '_compressed' + str(hyperparameters[5]) + '.png')

    # train logistic on class 0 and 6
    train_1_start = time()
    network_1 = network.Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, 1)
    logistic_predicted_weights = network_1.predict(logistic_train_1)
    train_1_end = time()
    train_1_time = train_1_end - train_1_start
    
    # train logistic on class 0 and 6 on compressed data
    train_1_start_compressed = time()
    network_1_compressed = network.Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, 1)
    logistic_predicted_weights_compressed = network_1_compressed.predict(logistic_train_1_compressed)
    train_1_end_compressed = time()
    train_1_time_compressed = train_1_end_compressed - train_1_start_compressed

    # print validation accuracy
    logistic_accuracy_1_val = np.min(network_1.fold_val_accuracy_history)
    print("Validation Accuracy of class 0 and 6: " + str(logistic_accuracy_1_val) + ".\n")

    logistic_accuracy_1_val_compressed = np.min(network_1_compressed.fold_val_accuracy_history)
    print("Validation Accuracy of class 0 and 6 compressed: " + str(logistic_accuracy_1_val_compressed) + ".\n")

    # print the test for logistic 1
    logistic_accuracy_1 = network_1.test(logistic_test_1)[1]
    print("Logistic Regression class 0 and class 6 testing data accuracy: " +  
        str(logistic_accuracy_1) + "\n")
    
    # print the test for logistic 1 compressed
    logistic_accuracy_1_compressed = network_1_compressed.test(logistic_test_1_compressed)[1]
    print("Logistic Regression class 0 and class 6 testing data accuracy (compressed via SVD): " +  
        str(logistic_accuracy_1_compressed) + "\n")
    
    # print the image compression ratio
    print("The compressed image is " + str(round(hyperparameters[5]*(28+28+1)/(28*28)*100,3)) + "% of the original image size.\n")
    
    # print the accuracy ratio
    print("The compressed testing accuracy is " + str(round(logistic_accuracy_1_compressed/logistic_accuracy_1 * 100, 3)) + 
          "% of original test set.\n")
    
    # print runtime of both set
    print("Original set runtime: " + str(train_1_time) + ".\n")
    print("Compressed set runtime: " + str(train_1_time_compressed) + ".\n")
    
    # print the runtime ratio
    print("Compressed set spent " + str(train_1_time_compressed/train_1_time*100) + "% of original's runtime.\n")


    # plot: validation and train set loss for logistic regression class 0 and 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot((network_1.epoch_val_loss_record/network_1.index_count)[1:], label = "Validation Loss Original Images")
    ax1.plot((network_1.epoch_train_loss_record/network_1.index_count)[1:], label = "Train Loss Original Images")
    
    ax2.plot((network_1_compressed.epoch_val_loss_record/network_1_compressed.index_count)[1:], label = "Validation Loss Compressed Images")
    ax2.plot((network_1_compressed.epoch_train_loss_record/network_1_compressed.index_count)[1:], label = "Train Loss Compressed Images")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("loss")
    fig.suptitle('Logistic Regression Train & Val loss Class 0 and 6 Averaged Over k-fold')
    ax1.set_title('Orginal Images')
    ax2.set_title('Compressed Images')
    legend = ax1.legend(loc='upper right')
    legend = ax2.legend(loc='upper right')
    # set up the x axis
    plt_threshold_1 = data.np.where(network_1.index_count[1:] == 0)[0]
    if len(plt_threshold_1) == 0:
        plt_threshold_1 = 101
    else:
        plt_threshold_1 = plt_threshold_1[0]
    ax1.set_xticks(data.np.arange(1, plt_threshold_1, 5))
    ax2.set_xticks(data.np.arange(1, plt_threshold_1, 5))
    plt.savefig("out_images/" + str(hyperparameters[2]) +"-" + str(hyperparameters[0]) + 'Logistic_0_6.png')
    
    
    
    
    # training dataset for logistic regression class 2 and class 6
    logistic_train_2 = data.np.where((data.load_data(True)[1] == 2) | (data.load_data(True)[1] == 6))
    logistic_train_2 = data.load_data(True)[0][logistic_train_2[0]], data.load_data(True)[1][logistic_train_2[0]]
    binary_6_to_1 = data.np.where(logistic_train_2[1] == 6, 1, logistic_train_2[1])
    binary_2_to_0 = data.np.where(binary_6_to_1 == 2, 0, binary_6_to_1)
    # compress via SVD
    compressed_train_2 = []
    for img in (logistic_train_2[0]):
        compressed_train_2.append(compress.compress_image(img.reshape(28,-1), hyperparameters[5]).reshape(28*28))
    compressed_train_2 = np.array(compressed_train_2)
    # original set
    logistic_train_2 = logistic_train_2[0], binary_2_to_0
    logistic_train_2 = data.append_bias(hyperparameters[3](logistic_train_2[0])[0]), logistic_train_2[1]
    logistic_train_2 = data.shuffle(logistic_train_2)
    # compressed set
    logistic_train_2_compressed = compressed_train_2, binary_2_to_0
    logistic_train_2_compressed = data.append_bias(
        hyperparameters[3](logistic_train_2_compressed[0])[0]), logistic_train_2_compressed[1]
    logistic_train_2_compressed = data.shuffle(logistic_train_2_compressed) 

    # testing dataset for logistic regression class 2 and class 6
    logistic_test_2 = data.np.where((data.load_data(False)[1] == 2) | (data.load_data(False)[1] == 6))
    logistic_test_2 = data.load_data(False)[0][logistic_test_2[0]], data.load_data(False)[1][logistic_test_2[0]]
    binary_6_to_1_2 = data.np.where(logistic_test_2[1] == 6, 1, logistic_test_2[1])
    binary_2_to_0_2 = data.np.where(binary_6_to_1_2 == 2, 0, binary_6_to_1_2)
    # compress via SVD
    compressed_test_2 = []
    for img in (logistic_test_2[0]):
        compressed_test_2.append(compress.compress_image(img.reshape(28,-1), hyperparameters[5]).reshape(28*28))
    compressed_test_2 = np.array(compressed_test_2)
    # original set
    logistic_test_2 = logistic_test_2[0], binary_2_to_0_2
    logistic_test_2 = data.append_bias(hyperparameters[3](logistic_test_2[0])[0]), logistic_test_2[1]
    # compressed set
    logistic_test_2_compressed = compressed_test_2, binary_2_to_0_2
    logistic_test_2_compressed = data.append_bias(
        hyperparameters[3](logistic_test_2_compressed[0])[0]), logistic_test_2_compressed[1]
    logistic_test_2_compressed = data.shuffle(logistic_test_2_compressed)

    # train logistic on class 2 and 6
    train_2_start = time()
    network_2 = network.Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, 1)
    logistic_predicted_weights_2 = network_2.predict(logistic_train_2)
    train_2_end = time()
    train_2_time = train_2_end - train_2_start
    
    # train logistic on class 2 and 6 on compressed data
    train_2_start_compressed = time()
    network_2_compressed = network.Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, 1)
    logistic_predicted_weights_compressed_2 = network_2_compressed.predict(logistic_train_2_compressed)
    train_2_end_compressed = time()
    train_2_time_compressed = train_2_end_compressed - train_2_start_compressed

    # print validation accuracy
    logistic_accuracy_2_val = np.min(network_2.fold_val_accuracy_history)
    print("Validation Accuracy of class 2 and 6: " + str(logistic_accuracy_2_val) + ".\n")

    logistic_accuracy_2_val_compressed = np.min(network_2_compressed.fold_val_accuracy_history)
    print("Validation Accuracy of class 2 and 6 compressed: " + str(logistic_accuracy_2_val_compressed) + ".\n")

    # print the test for logistic 2
    logistic_accuracy_2 = network_2.test(logistic_test_2)[1]
    print("Logistic Regression class 2 and class 6 testing data accuracy: " +  
        str(logistic_accuracy_2) + "\n")
    
    # print the test for logistic 2 compressed
    logistic_accuracy_2_compressed = network_2_compressed.test(logistic_test_2_compressed)[1]
    print("Logistic Regression class 2 and class 6 testing data accuracy (compressed via SVD): " +  
        str(logistic_accuracy_2_compressed) + "\n")
    
    # print the image compression ratio
    print("The compressed image is " + str(round(hyperparameters[5]*(28+28+1)/(28*28)*100,3)) + "% of the original image size.\n")
    
    # print the accuracy ratio
    print("The testing accuracy is " + str(round(logistic_accuracy_2_compressed/logistic_accuracy_2, 3)) + 
          "% of uncompressed test set.\n")
    
    # print runtime of both set
    print("Original set runtime: " + str(train_2_time) + ".\n")
    print("Compressed set runtime: " + str(train_2_time_compressed) + ".\n")
    
    # print the runtime ratio
    print("Compressed set spent " + str(train_2_time_compressed/train_2_time*100) + "% of original's runtime.\n")


    # plot: validation and train set loss for logistic regression class 2 and 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot((network_2.epoch_val_loss_record/network_2.index_count)[1:], label = "Validation Loss Original Images")
    ax1.plot((network_2.epoch_train_loss_record/network_2.index_count)[1:], label = "Train Loss Original Images")
    
    ax2.plot((network_2_compressed.epoch_val_loss_record/network_2_compressed.index_count)[1:], label = "Validation Loss Compressed Images")
    ax2.plot((network_2_compressed.epoch_train_loss_record/network_2_compressed.index_count)[1:], label = "Train Loss Compressed Images")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("loss")
    fig.suptitle('Logistic Regression Train & Val loss Class 2 and 6 Averaged Over k-fold')
    ax1.set_title('Orginal Images')
    ax2.set_title('Compressed Images')
    legend = ax1.legend(loc='upper right')
    legend = ax2.legend(loc='upper right')
    # set up the x axis
    plt_threshold_2, x_2 = data.np.where(network_2.index_count[1:] == 0)[0], 1
    if len(plt_threshold_2) == 0:
        plt_threshold_2, x_2 = 101, 5
    else:
        plt_threshold_2, x_2 = data.np.where(network_2.index_count[1:] == 0)[0][0], 3
    ax1.set_xticks(data.np.arange(1, plt_threshold_2, x_2))
    ax2.set_xticks(data.np.arange(1, plt_threshold_2, x_2))
    plt.savefig("out_images/" + str(hyperparameters[2]) +"-" + str(hyperparameters[0]) + 'Logistic_2_6.png')

    print("All Trainings Done! Please check the saved images in `out_images` folder in the same directory:)")
    return

parser = argparse.ArgumentParser(description = 'Experiment')
parser.add_argument('--batch-size', type = int, default = 32,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', type = float, default = 0.001,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', dest = 'normalization', action='store_const',
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', type = int, default = 10,
        help = 'number of folds for cross-validation')
parser.add_argument('--svd-limit', type = int, default = 3,
        help = 'singular value decomposition limit')

hyperparameters = parser.parse_args()
print('Experiment Starts!')
main(hyperparameters)
