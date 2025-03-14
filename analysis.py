#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 01:32:23 2025

@author: tanmaygandhi
"""

import ml_full

# prep training data and build ml model
my_train_model = ml_full.process_data(inputdata = '0_raw/census_income_learn.csv', metadata = '0_raw/census_col_labels.csv', out_dir = '1_results', label = 'train', trainingrun=True)

# prep test data
my_test_data = ml_full.process_data(inputdata = '0_raw/census_income_test.csv', metadata = '0_raw/census_col_labels.csv', out_dir = '1_results', label = 'test', trainingrun=False)

# run prediction on test data
pred_results = ml_full.pred_data(ml_model=my_train_model, test_data=my_test_data, out_dir='1_results/')

# visualize model training and prediction performance
ml_full.viz_results(ml_model=my_train_model, y_true=pred_results['target'], y_scores=pred_results['prediction_score'], out_dir='1_results')
