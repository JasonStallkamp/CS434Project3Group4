import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, print_dictonary, calc_f1
from tree import DecisionTreeClassifier, RandomForestClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	parser.add_argument('--depth_test', default=0, type=int)
	parser.add_argument('--depth_test_min', default=1, type=int)
	parser.add_argument('--depth_test_max', default=25, type=int)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	print_dictonary(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
	print('#Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=20)

	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)

	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

def decision_tree_testing_depth(x_train, y_train, x_test, y_test, min, max):
	print('#Decision Tree Depth Testing\n\n')
	accuracyTrain = np.zeros(max - min)
	accuracyTest = np.zeros(max - min)
	f1Train = np.zeros(max - min)
	f1Test = np.zeros(max - min)
	depths = np.arange(min, max)
	index = 0
	for depth in depths:
		clf = DecisionTreeClassifier(max_depth=depth)
		clf.fit(x_train, y_train)
		preds_train = clf.predict(x_train)
		preds_test = clf.predict(x_test)
		accuracyTrain[index] = accuracy_score(preds_train, y_train)
		accuracyTest[index] = accuracy_score(preds_test, y_test)
		preds = clf.predict(x_test)
		f1Test[index] = calc_f1(preds_train, y_train)
		f1Train[index] = calc_f1(preds_test, y_test)
		index += 1
	f1 = plt.figure(1)
	plt.plot(depths, accuracyTrain)
	plt.plot(depths, accuracyTest)
	plt.title("accuracy vs number of trees")
	plt.ylabel("Accuracy")
	plt.xlabel("Depth")
	plt.legend(['Training Accuracy', 'Testing Accuracy'])
	f1.show()

	f2 = plt.figure(2)
	plt.plot(depths, f1Train)
	plt.plot(depths, f1Test)
	plt.title("F1 vs number of trees")
	plt.ylabel("F1")
	plt.xlabel("Depth")
	plt.legend(['Training F1', 'Testing F1'])
	plt.show()


def random_forest_testing(x_train, y_train, x_test, y_test):
	print('#Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))



###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	if args.county_dict == 1:
		county_info(args)
	if args.decision_tree == 1:
		decision_tree_testing(x_train, y_train, x_test, y_test)
	if args.depth_test == 1:
		decision_tree_testing_depth(x_train, y_train, x_test, y_test, args.depth_test_min, args.depth_test_max + 1)
	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)

	print('Done')
	
	





