import csv
import random
import math
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


_author_ = 'Aman Chopra'
_description_ = 'Define classes and methods to recognize and classify images from the MNIST dataset using KNN.'
_puid_ = '0030974520'


# class MNISTPoint:
#     def __init__(self, id, class_label, x, y):
#         self.id = id
#         self.class_label = class_label
#         self.x = x
#         self.y = y

def most_common_element(list):
    return max(list, key=list.count)


class MyKNN:
    def readData(self, filename):
        training_ids = []
        testing_ids = []
        with open(filename, 'rb') as csvfile:
            temp_dataset = list(csv.reader(csvfile))
            dataset = {}
            for data in temp_dataset:
                dataset[int(data[0])] = [int(data[1]), float(data[2]), float(data[3])]  # values include list of class label, x-coord, y-coord
            random.shuffle(temp_dataset)
            for i in range(int(0.8 * len(dataset))):  # 80% of dataset (16,000/20,000) will be used in the training set
                training_ids.append(int(temp_dataset[i][0]))
            for i in range(int(0.8 * len(dataset)), len(dataset)):  # 20% of dataset (4,000/20,000) will be used in testing set
                testing_ids.append(int(temp_dataset[i][0]))
            training_ids.sort()
            testing_ids.sort()
        return dataset, training_ids, testing_ids

    def split_dataset_even_odd(self, dataset, class_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        training_ids_even = [x for x in dataset if dataset[x][0] in class_labels and x % 2 == 0]
        testing_ids_odd = [x for x in dataset if dataset[x][0] in class_labels and x % 2 != 0]
        # for id in dataset:
        #     if id % 2 == 0:
        #         training_ids_even.append(id)
        #     else:
        #         testing_ids_odd.append(id)

        training_ids_even.sort()
        testing_ids_odd.sort()
        return training_ids_even, testing_ids_odd

    def classify(self, k, parsed_data, training_ids, test_ids):
        if k > len(training_ids):
            raise Exception('K too large!')  # k cannot exceed size of training ids
        euclidean_distance = {}
        for test_id in test_ids:  # stores euclidean distance for each training point and test point
            euclidean_distance[test_id] = []
            for training_id in training_ids:
                euclidean_distance[test_id].append([training_id, math.sqrt(math.pow(parsed_data[test_id][1] - parsed_data[training_id][1], 2) + math.pow(parsed_data[test_id][2] - parsed_data[training_id][2], 2))])
        for id in euclidean_distance:  # sorts dict by euclidean distance
            euclidean_distance[id].sort(key=lambda x: x[1])
        class_labels = []
        for id in euclidean_distance:  # stores all class labels for every test id
            neighbors = []
            for i in range(k):
                neighbors.append(euclidean_distance[id][i][0])
            classes = []
            for neighbor in neighbors:
                classes.append(parsed_data[neighbor][0])
            class_labels.append([id, classes])
        class_labels.sort(key=lambda x: x[0])  # sorts class labels for all ids so prediction list corresponds w/ test id list
        predictions = []
        for id in class_labels:
            predictions.append(most_common_element(id[1]))
        return predictions

    def evaluate(self, parsed_data, test_ids, predictions, class_labels=[]):
        # classifications = OrderedDict()
        # for i in range(len(test_ids)):
        #     classifications[test_ids[i]] = [parsed_data[test_ids[i]][0], predictions[i]]
        classifications = []
        for i in range(len(
                test_ids)):  # creates classifications array that contains arrays consisting actual and predicted class labels
            classifications.append([parsed_data[test_ids[i]][0], predictions[i]])
        classifications.sort(key=lambda x: x[0])
        # classifications_vals = classifications.values()
        # classifications_vals.sort(key=lambda x: x[0])
        confusion_matrix = np.zeros((10, 10))  # initialize np array to zeros
        classifications_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [],
                                9: []}  # dict containing predicted classes for each class
        for classification in classifications:
            if classification[0] == 0:
                classifications_dict[0].append(classification[1])
            elif classification[0] == 1:
                classifications_dict[1].append(classification[1])
            elif classification[0] == 2:
                classifications_dict[2].append(classification[1])
            elif classification[0] == 3:
                classifications_dict[3].append(classification[1])
            elif classification[0] == 4:
                classifications_dict[4].append(classification[1])
            elif classification[0] == 5:
                classifications_dict[5].append(classification[1])
            elif classification[0] == 6:
                classifications_dict[6].append(classification[1])
            elif classification[0] == 7:
                classifications_dict[7].append(classification[1])
            elif classification[0] == 8:
                classifications_dict[8].append(classification[1])
            elif classification[0] == 9:
                classifications_dict[9].append(classification[1])
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                confusion_matrix[i, j] = classifications_dict[j].count(i)
        f1_scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for j in range(confusion_matrix.shape[1]):
            tp = confusion_matrix[j, j]
            fn = sum(confusion_matrix[..., j]) - tp
            fp = sum(confusion_matrix[j, ...]) - tp
            tn = confusion_matrix.sum() - tp - fn - fp
            if tp == 0 and fp == 0 and fn == 0:
                continue
            f1_scores[j] = float((2 * tp) / (2 * tp + fp + fn))
        avg_f1 = sum(f1_scores.values()) / (10 - len(class_labels))
        confusion_matrix = np.delete(confusion_matrix, tuple(class_labels), axis=0)
        confusion_matrix = np.delete(confusion_matrix, tuple(class_labels), axis=1)
        np.set_printoptions(suppress=True)

        return confusion_matrix, avg_f1

    def constructPlot(self, title, dataset=None, k_points=None, f1_avg_points=None, labels=None):
        if k_points is None and f1_avg_points is None:
            points = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            for entries in dataset.values():
                if entries[0] == 0:
                    points[0].append([entries[1], entries[2]])
                    zero = plt.scatter(entries[1], entries[2], c='b', marker='x', linewidths=2)
                elif entries[0] == 1:
                    points[1].append([entries[1], entries[2]])
                    one = plt.scatter(entries[1], entries[2], c='g')
                elif entries[0] == 2:
                    points[2].append([entries[1], entries[2]])
                    two = plt.scatter(entries[1], entries[2], c='r', marker='x', linewidths=2)
                elif entries[0] == 3:
                    points[3].append([entries[1], entries[2]])
                    three = plt.scatter(entries[1], entries[2], c='c')
                elif entries[0] == 4:
                    points[4].append([entries[1], entries[2]])
                    four = plt.scatter(entries[1], entries[2], c='m')
                elif entries[0] == 5:
                    points[5].append([entries[1], entries[2]])
                    five = plt.scatter(entries[1], entries[2], c='y', marker='x', linewidths=2)
                elif entries[0] == 6:
                    points[6].append([entries[1], entries[2]])
                    six = plt.scatter(entries[1], entries[2], c='k')
                elif entries[0] == 7:
                    points[7].append([entries[1], entries[2]])
                    seven = plt.scatter(entries[1], entries[2], c='k', marker='x', linewidths=2)
                elif entries[0] == 8:
                    points[8].append([entries[1], entries[2]])
                    eight = plt.scatter(entries[1], entries[2], c='0.45')
                elif entries[0] == 9:
                    points[9].append([entries[1], entries[2]])
                    nine = plt.scatter(entries[1], entries[2], c='b')
            try:  # exception handling in case that dataset omits certain classes (1, 9, 8, or 5)
                plt.legend((zero, one, two, three, four, five, six, seven, eight, nine),
                           ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
            except UnboundLocalError:
                if len(points[8]) == 0:
                    plt.legend((one, nine),
                               ('1', '9'))
                else:
                    plt.legend((one, nine, eight, five),
                               ('1', '9', '8', '5'))
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            plt.plot(k_points, f1_avg_points, marker='o', c='k')
            for x, y, label in zip(k_points, f1_avg_points, labels):
                plt.annotate("  " + label, xy=(x, y))

            plt.title(title)
            plt.xlabel('k value')
            plt.ylabel('F1 score')
            plt.show()


# def main():
#     knn_alg = MyKNN()
#     dataset, training_ids, testing_ids = knn_alg.readData('test-data.csv')
#
#     # for report
#     even_training_ids_full, odd_testing_ids_full = knn_alg.split_dataset_even_odd(dataset)
#     even_training_ids_1985, odd_testing_ids_1985 = knn_alg.split_dataset_even_odd(dataset, class_labels=[1, 9, 8, 5])
#     even_training_ids_19, odd_testing_ids_19 = knn_alg.split_dataset_even_odd(dataset, class_labels=[1, 9])
#     subset_one_nine_eight_five = dataset.copy()
#     subset_one_nine = dataset.copy()
#     for id in subset_one_nine_eight_five.keys():
#         if subset_one_nine_eight_five[id][0] != 1 and subset_one_nine_eight_five[id][0] != 9 and subset_one_nine_eight_five[id][0] != 8 and subset_one_nine_eight_five[id][0] != 5:
#             del subset_one_nine_eight_five[id]
#     for id in subset_one_nine.keys():
#         if subset_one_nine[id][0] != 1 and subset_one_nine[id][0] != 9:
#             del subset_one_nine[id]
#     knn_alg.constructPlot('Full Dataset', dataset=dataset)
#     knn_alg.constructPlot('Subset w/ 1, 9, 8, and 5', dataset=subset_one_nine_eight_five)
#     knn_alg.constructPlot('Subset w/ 1 and 9', dataset=subset_one_nine)
#
#     cases = [[dataset, [], even_training_ids_full, odd_testing_ids_full], [subset_one_nine_eight_five, [0, 2, 3, 4, 6, 7], even_training_ids_1985, odd_testing_ids_1985], [subset_one_nine_eight_five, [0, 2, 3, 4, 5, 6, 7, 8], even_training_ids_19, odd_testing_ids_19]]
#     for case in cases:
#         confusion_matrix_k1, avg_f1_k1 = knn_alg.evaluate(case[0], case[3],
#                                                           knn_alg.classify(1, case[0], case[2],
#                                                                            case[3]), class_labels=case[1])
#         confusion_matrix_k5, avg_f1_k5 = knn_alg.evaluate(case[0], case[3],
#                                                           knn_alg.classify(5, case[0], case[2],
#                                                                            case[3]), class_labels=case[1])
#         confusion_matrix_k15, avg_f1_k15 = knn_alg.evaluate(case[0], case[3],
#                                                             knn_alg.classify(15, case[0], case[2],
#                                                                              case[3]), class_labels=case[1])
#         confusion_matrix_k31, avg_f1_k31 = knn_alg.evaluate(case[0], case[3],
#                                                             knn_alg.classify(31, case[0], case[2],
#                                                                              case[3]), class_labels=case[1])
#         confusion_matrix_kq, avg_f1_kq = knn_alg.evaluate(case[0], case[3],
#                                                           knn_alg.classify(len(case[2]), case[0],
#                                                                            case[2], case[3]), class_labels=case[1])
#         k_points = [1, 5, 15, 31]
#         f1_points = [avg_f1_k1, avg_f1_k5, avg_f1_k15, avg_f1_k31]
#         labels = ['k = 1', 'k = 5', 'k = 15', 'k = 31']
#         knn_alg.constructPlot('F1 score vs k', k_points=k_points, f1_avg_points=f1_points, labels=labels)
#         print 'Confusion Matrix When k = 1:'
#         print confusion_matrix_k1
#         print 'Average F1 Score When k = 1'
#         print avg_f1_k1
#         print
#         print 'Confusion Matrix When k = 5:'
#         print confusion_matrix_k5
#         print 'Average F1 Score When k = 5'
#         print avg_f1_k5
#         print
#         print 'Confusion Matrix When k = 15:'
#         print confusion_matrix_k15
#         print 'Average F1 Score When k = 15'
#         print avg_f1_k15
#         print
#         print 'Confusion Matrix When k = 31:'
#         print confusion_matrix_k31
#         print 'Average F1 Score When k = 31'
#         print avg_f1_k31
#         print
#         print 'Confusion Matrix When k = q:'
#         print confusion_matrix_kq
#         print 'Average F1 Score When k = q'
#         print avg_f1_kq
#
#
# if __name__ == '__main__':
#     main()





