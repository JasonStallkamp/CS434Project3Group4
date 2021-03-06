import numpy as np
import math


class Node:
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier:
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	# take in features X and labels y
	# build a tree
	def fit(self, X, y, feature_idx=None, weights=None):
		self.num_classes = len(set(y))
		self.class_labels = list(set(y))
		if feature_idx is None:
			self.features_idx = np.arange(0, X.shape[1])
		else:
			self.features_idx = feature_idx

		if weights is None:
			weights = np.ones(X.shape[0])
		self.root = self.build_tree(X, y, 1, np.asarray(weights))

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth, weights):

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None
		best_left_weight = None
		best_right_weight = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in self.class_labels]
		prediction = self.class_labels[np.argmax(num_samples_per_class)]

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y, left_weights, right_weights = self.check_split(X, y, feature, split, weights)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
						best_left_weight = left_weights
						best_right_weight = right_weights
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth+1, best_left_weight)
			right_tree = self.build_tree(best_right_X, best_right_y, depth+1, best_right_weight)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split, weights):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)

		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		left_weights = weights[left_idx]
		right_weights = weights[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y, weights, left_weights, right_weights)
		return gain, left_X, right_X, left_y, right_y, left_weights, right_weights

	def calculate_gini_gain(self, y, left_y, right_y, weights, left_weights, right_weights):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			x = np.sum((y == self.class_labels[0]) * weights)
			cp = np.sum((y == self.class_labels[0]) * weights)
			cn = np.sum((y == self.class_labels[1]) * weights)
			clp = np.sum((left_y == self.class_labels[0]) * left_weights)
			cln = np.sum((left_y == self.class_labels[1]) * left_weights)
			crp = np.sum((right_y == self.class_labels[0]) * right_weights)
			crn = np.sum((right_y == self.class_labels[1]) * right_weights)
			ul = 1 - pow(clp / (clp + cln), 2) - pow(cln / (clp + cln), 2)
			ur = 1 - pow(crp / (crp + crn), 2) - pow(crn / (crp + crn), 2)
			ua = 1 - pow(cp / (cp + cn), 2) - pow(cn / (cp + cn), 2)
			pl = (clp + cln) / (cp + cn)
			pr = (crp + crn) / (cp + cn)
			gain = ua - (pl * ul) - (pr * ur)
			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0


class RandomForestClassifier:
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		bagged_features = self.bag_features(X)
		print('Fitting Random Forest...\n')
		self.trees = []
		for i in range(self.n_trees):
			newTree = DecisionTreeClassifier(max_depth=self.max_depth)
			newTree.fit(bagged_X[i], bagged_y[i], feature_idx=bagged_features[i])
			self.trees.append(newTree)

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		for i in range(self.n_trees):
			rands = np.random.random_integers(0, high=len(X) - 1, size=len(X))
			bagged_X.append(X[rands])
			bagged_y.append(y[rands])

		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)

	def bag_features(self, X):
		#bag features
		features = []
		for i in range(self.n_trees):
			features_idx = np.arange(0, X.shape[1])
			features.append(np.take(np.random.permutation(features_idx), np.arange(0, self.max_features)))
		return np.array(features)

	def predict(self, X):
		preds = np.zeros(X.shape[0])
		for i in range(self.n_trees):
			pre_pred = np.asarray(self.trees[i].predict(X))
			pre_pred = pre_pred * 2 - 1
			preds = preds + pre_pred
		preds = ((preds / abs(preds)) + 1) / 2
		##################
		# YOUR CODE HERE #
		##################
		return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier:
	def __init__(self, num_trees, max_depth):
		self.num_trees = num_trees
		self.max_depth = max_depth
		self.trees = []
		self.alphas = []

	def train(self, X, y):
		# Initialize D1(i) = 1/N for all i from 1 to N
		weights = []
		self.trees = []
		self.alphas = []
		weights.append(np.ones(X.shape[0]) / X.shape[0])

		for i in range(0, self.num_trees):
			# create new tree
			newTree = DecisionTreeClassifier(max_depth=self.max_depth)
			newTree.fit(X, y, weights=weights[i])
			self.trees.append(newTree)

			# Calc error rate
			error = 1 - newTree.accuracy_score(X, y)
			alpha = .5 * math.log((1-error)/error, math.e)
			self.alphas.append(alpha)
			predict = newTree.predict(X)

			# Calc new weight factor
			new_weight = weights[i] * np.exp(alpha * ((predict != y) * 2 - 1))
			normalize_factor = sum(new_weight)
			weights.append( new_weight / normalize_factor)

	def predict(self, X):
		y = np.zeros(X.shape[0])
		for i in range(0, len(self.trees)):
			y = y + np.asarray(self.trees[i].predict(X)) * self.alphas[i]
		return np.sign(y)









