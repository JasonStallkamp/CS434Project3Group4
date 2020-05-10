import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, print_dictonary, calc_f1, shiftData
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=0, type=int)
	parser.add_argument('--random_forest', default=0, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	parser.add_argument('--depth_test', default=0, type=int)
	parser.add_argument('--depth_test_min', default=1, type=int)
	parser.add_argument('--depth_test_max', default=25, type=int)

	parser.add_argument('--forest_test_trees', default=0, type=int)
	parser.add_argument('--forest_test_max_features', default=0, type=int)
	parser.add_argument('--forest_random', default=0, type=int)
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
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=51)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))


def random_forest_testing_max_trees(x_train, y_train, x_test, y_test):
	print('#Random Forest Number of Trees\n\n')
	accuracy_training = []
	accuracy_testing = []
	f1_testing = []
	f1_training = []
	num_trees = []
	for max_trees in range(10, 210, 10):
		rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=max_trees)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		num_trees.append(max_trees)
		accuracy_training.append(accuracy_score(preds_train, y_train))
		accuracy_testing.append(accuracy_score(preds_test, y_test))
		f1_training.append(calc_f1(preds_train, y_train))
		f1_testing.append(calc_f1(preds_test, y_test))

	f1 = plt.figure(1)
	plt.plot(num_trees, accuracy_training)
	plt.plot(num_trees, accuracy_testing)
	plt.title("accuracy vs number of trees")
	plt.ylabel("Accuracy")
	plt.xlabel("Number of trees")
	plt.legend(['Training Accuracy', 'Testing Accuracy'])
	f1.show()

	f2 = plt.figure(2)
	plt.plot(num_trees, f1_training)
	plt.plot(num_trees, f1_testing)
	plt.title("F1 vs number of trees")
	plt.ylabel("F1")
	plt.xlabel("number of trees")
	plt.legend(['Training F1', 'Testing F1'])
	plt.show()


def random_forest_testing_max_features(x_train, y_train, x_test, y_test):
	print('#Random Forest Number of Trees\n\n')
	accuracy_training = []
	accuracy_testing = []
	f1_testing = []
	f1_training = []
	features = []
	for max_features in [1, 2, 5, 8, 10, 20, 25, 35, 50]:
		rclf = RandomForestClassifier(max_depth=7, max_features=max_features, n_trees=50)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		features.append(max_features)
		accuracy_training.append(accuracy_score(preds_train, y_train))
		accuracy_testing.append(accuracy_score(preds_test, y_test))
		f1_training.append(calc_f1(preds_train, y_train))
		f1_testing.append(calc_f1(preds_test, y_test))

	f1 = plt.figure(1)
	plt.plot(features, accuracy_training)
	plt.plot(features, accuracy_testing)
	plt.title("Accuracy vs Max Features")
	plt.ylabel("Accuracy")
	plt.xlabel("Max Features")
	plt.legend(['Training Accuracy', 'Testing Accuracy'])
	f1.show()

	f2 = plt.figure(2)
	plt.plot(features, f1_training)
	plt.plot(features, f1_testing)
	plt.title("F1 vs Max Features")
	plt.ylabel("F1")
	plt.xlabel("Max Features")
	plt.legend(['Training F1', 'Testing F1'])
	plt.show()


def random_forsest_random_seed(x_train, y_train, x_test, y_test, count):
	print('#Random Forest Number of Trees\n\n')
	accuracy_training = []
	accuracy_testing = []
	f1_testing = []
	f1_training = []
	features = []
	for i in range(0, count):
		rclf = RandomForestClassifier(max_depth=7, max_features=25, n_trees=151)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		features.append(i)
		accuracy_training.append(accuracy_score(preds_train, y_train))
		accuracy_testing.append(accuracy_score(preds_test, y_test))
		f1_training.append(calc_f1(preds_train, y_train))
		f1_testing.append(calc_f1(preds_test, y_test))

	f1 = plt.figure(1)
	plt.plot(features, accuracy_training)
	plt.plot(features, accuracy_testing)
	plt.title("Accuracy vs Seed")
	plt.ylabel("Accuracy")
	plt.xlabel("Seed Index")
	plt.legend(['Training Accuracy', 'Testing Accuracy'])
	f1.show()

	f2 = plt.figure(2)
	plt.plot(features, f1_training)
	plt.plot(features, f1_testing)
	plt.title("F1 vs Seed")
	plt.ylabel("F1")
	plt.xlabel("Seed Index")
	plt.legend(['Training F1', 'Testing F1'])
	plt.show()


def ada_boost(x_train, y_train, x_test, y_test):
	print('#ADA Boost Testing\n\n')
	shift_y_train = shiftData(y_train)
	shift_y_test = shiftData(y_test)

	train_accuracy = []
	test_accuracy = []
	train_f1 = []
	test_f1 = []
	parameter = []

	for num_trees in range(10, 100, 10):
		print("Testing with ", num_trees, " trees")
		ada = AdaBoostClassifier(num_trees, 1)
		ada.train(x_train, shift_y_train)

		preds_train = ada.predict(x_train)
		preds_test = ada.predict(x_test)
		parameter.append(num_trees)
		train_accuracy.append(accuracy_score(preds_train, shift_y_train))
		test_accuracy.append(accuracy_score(preds_test, shift_y_test))
		train_f1.append(calc_f1(preds_train, shift_y_train))
		test_f1.append(calc_f1(preds_test, shift_y_test))

	f1 = plt.figure(1)
	plt.plot(parameter, train_accuracy)
	plt.plot(parameter, test_accuracy)
	plt.title("ADA Boost Accuracy vs Number of Trees")
	plt.ylabel("Accuracy")
	plt.xlabel("Number of Trees")
	plt.legend(['Training Accuracy', 'Testing Accuracy'])
	f1.show()

	f2 = plt.figure(2)
	plt.plot(parameter, train_f1)
	plt.plot(parameter, test_f1)
	plt.title("ADA Boost F1 vs Number of Trees")
	plt.ylabel("F1")
	plt.xlabel("Number of Trees")
	plt.legend(['Training F1', 'Testing F1'])
	plt.show()

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
	if args.forest_test_trees == 1:
		random_forest_testing_max_trees(x_train,y_train, x_test, y_test)
	if args.forest_test_max_features == 1:
		random_forest_testing_max_features(x_train, y_train, x_test, y_test)
	if args.forest_random != 0:
		random_forsest_random_seed(x_train, y_train, x_test, y_test, args.forest_random)
	if args.ada_boost == 1:
		ada_boost(x_train, y_train, x_test, y_test)
	print('Done')
	
	





