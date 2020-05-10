To run the code run python3 main.py
Additional arguments are


--decision_tree 1                   for the decision tree
--depth_test 1                      to run a depth test for the decision tree
--depth_test_min value              will set the depth test minumum value (inclusive)
--depth_test_max value              will set teh depth test maximum value (inclusive)

--random_forest 1                   to run the random forest
--forest_test_trees value           to set the number of trees in the random forest
--forest_test_max_features value    set the maximum features for the random seeded forest test
--forest_random value               to run an array of random seeded forests. it will run value times

--ada_boost 1                       to run the ada boost