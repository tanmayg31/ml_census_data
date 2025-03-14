#!/usr/bin/env python
# coding: utf-8

# # Modeling Census Tract Data

# * Census data is provided in [census_income_learn.csv](0_raw/census_income_learn.csv)
# * Raw metadata is provided in [census_income_metadata.txt](0_raw/census_income_metadata.txt)
# * Census data does not contain column labels
# * Manually extracted and cleaned column names from raw metadata file above to create new file containing column names for the census dataset: [census_col_labels.csv](0_raw/census_col_labels.csv)


# set flags
FULL_VERBOSE = True
MAKE_PLOTS = True
SAVE_PLOTS = True

# import libraries
import pandas as pd
import numpy as np
import re
import os
from pycaret.classification import *

if MAKE_PLOTS:
    import matplotlib.pyplot as plt
    import seaborn as sns

if SAVE_PLOTS:
    from matplotlib.backends.backend_pdf import PdfPages

# define a custom print function
def custom_print(text):
    print("#" * 80)
    print(text)
    print("#" * 80)
    print("\n")

# define function to load and clean up data
def process_data(inputdata, metadata, out_dir, label, trainingrun = True):
    # read in the data
    census_df = pd.read_csv(inputdata, header=None)
    col_labels = pd.read_csv(metadata)

    # create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # print number of rows and columns in the census dataset
    custom_print(f"Number of rows (raw): {census_df.shape[0]}")
    custom_print(f"Number of columns (raw): {census_df.shape[1]}")

    # show what the data looks like
    custom_print(f"Raw data:\n{census_df.head()}")

    # add col labels to census df
    census_df.columns = col_labels.iloc[:,1].tolist()

    # show what the data looks like
    custom_print(f"Raw data with col labels:\n{census_df.head()}")

    ## Exploratory Data Analysis I
    if FULL_VERBOSE:
        # info about all variables
        print(census_df.info())

        # summary of numerical columns
        census_df.select_dtypes(include=['number']).describe()

        # summary of categorical columns
        categorical_columns = census_df.select_dtypes(include=['object','category']).columns
        for col in categorical_columns:
            custom_print(f"Summary of {col}:\n{census_df[col].value_counts(dropna=False)}")

    ## Preprocessing
    # following terms are used to describe NAs in the dataset
    na_list = ['Not in universe', 'Not in universe or children', 'Do not know', '?', 'Not identifiable', 'Not in universe under 1 year old']

    # replace these terms with NA by rereading in the census data
    census_df_clean = pd.read_csv(inputdata, header=None, names=col_labels.iloc[:,1].tolist(), na_values=na_list, keep_default_na=True, skipinitialspace=True)

    # for several vars, 0 represents a missing value, hence replace 0 with NA in those vars
    cols_to_convert = ['wage_per_hour','capital_gains','capital_losses','dividends_from_stocks','num_persons_worked_for_employer','own_business_or_self_employed','weeks_worked_in_year']
    census_df_clean[cols_to_convert] = census_df_clean[cols_to_convert].replace(0, np.nan)

    # for "own_business_or_self_employed", replace 1 with "Yes" and 2 with "No"
    census_df_clean['own_business_or_self_employed'] = census_df_clean['own_business_or_self_employed'].replace({1: 'Yes', 2: 'No'})

    # column "year" only has 2 values: 94 and 95. Convert this to a categorical var
    census_df_clean['year'] = census_df_clean['year'].astype('category')

    custom_print(f"Shape before dropping rows with 70% or more NAs: {census_df_clean.shape}")

    # drop rows with 70% or more NAs
    census_df_clean = census_df_clean.dropna(thresh=int(0.3 * census_df_clean.shape[1]))

    custom_print(f"Shape after dropping rows with 70% or more NAs: {census_df_clean.shape}")

    # create new var from education by combining levels to create new classes "NoDegree", "Associates", "Bachelors", "Masters", "Doctorate", "ProfSchool"
    census_df_clean['new_education'] = census_df_clean['education']
    my_flag = census_df_clean['new_education'].str.contains('no degree|grade|children|High school', case=False, na=False)
    census_df_clean.loc[my_flag,'new_education'] = 'NoDegree'
    census_df_clean['new_education'] = census_df_clean['new_education'].replace({
        'Associates degree-occup /vocational': 'Associates',
        'Associates degree-academic program': 'Associates',
        'Bachelors degree(BA AB BS)': 'Bachelors',
        'Masters degree(MA MS MEng MEd MSW MBA)': 'Masters',
        'Doctorate degree(PhD EdD)': 'Doctorate',
        'Prof school degree (MD DDS DVM LLB JD)': 'ProfSchool'
    })

    # create new var from class of worker by combining private + self-employed and combining federal + local + state government
    census_df_clean['new_class_of_worker'] = census_df_clean['class_of_worker'].replace({
        'Private': 'Private',
        'Self-employed-not incorporated': 'Private',
        'Self-employed-incorporated': 'Private',
        'Federal government': 'Government',
        'Local government': 'Government',
        'State government': 'Government',
        'Never worked': pd.NA,
        'Without pay': pd.NA
    })

    # create new var called "moved" from "migration_code-move_within_reg" by converting "Nonmover" to "No" and all other values to "Yes"
    census_df_clean['moved'] = census_df_clean['migration_code-move_within_reg'].apply(lambda x: 'No' if x == 'Nonmover' else ('Yes' if pd.notna(x) else pd.NA))

    # create new var from family_members_under_18 called "both_parents_present" by converting "Both parents present" to "Yes" and all other values to "No"
    census_df_clean['both_parents_present'] = census_df_clean['family_members_under_18'].replace({
        'Both parents present': 'Yes',
        'Mother only present': 'No',
        'Father only present': 'No',
        'Neither parent present': 'No'
    })

    # create new var called "american_father" from "country_of_birth_father" by converting "United-States" to "Yes" and all other values to "No"
    census_df_clean['american_father'] = census_df_clean['country_of_birth_father'].apply(lambda x: 'Yes' if x == 'United-States' else ('No' if pd.notna(x) else pd.NA))

    # create new var called "american_mother" from "country_of_birth_mother" by converting "United-States" to "Yes" and all other values to "No"
    census_df_clean['american_mother'] = census_df_clean['country_of_birth_mother'].apply(lambda x: 'Yes' if x == 'United-States' else ('No' if pd.notna(x) else pd.NA))

    # for the target variable, replace "- 50000." with "lessthan50k" and "50000+." with "morethan50k"
    census_df_clean['target'] = census_df_clean['target'].replace({'- 50000.': 'lessthan50k', '50000+.': 'morethan50k'})

    # apply one hot encoding with preserved NAs to all discrete variables
    discrete_vars = ['new_education','race','hispanic_origin','region_of_previous_residence']
    temp_df = pd.DataFrame(index=census_df_clean.index)

    for col in discrete_vars:
        # create flag for NA values
        nan_flag = census_df_clean[col].isna()

        # create dummies for this column
        dummies = pd.get_dummies(census_df_clean[col], drop_first=True, dtype=float, prefix=col)

        # use NA flag to add NAs to all dummy columns
        for dummy_col in dummies.columns:
            dummies.loc[nan_flag, dummy_col] = np.nan

        # append to temp_df
        temp_df = pd.concat([temp_df, dummies], axis=1)

    # combine original columns with one hot encoded columns
    census_df_clean = pd.concat([census_df_clean.drop(discrete_vars, axis=1), temp_df], axis=1)

    # drop column "instance_weight" since the metadata file mentions that this variable is not to be used for modeling
    census_df_clean = census_df_clean.drop(columns=['instance_weight'])

    if FULL_VERBOSE:
        # info about all variables
        print(census_df_clean.info())

        # updated summary of numerical columns
        census_df_clean.select_dtypes(include=['number']).describe()

        # updated summary of categorical columns
        categorical_columns = census_df_clean.select_dtypes(include=['object','category']).columns
        for col in categorical_columns:
            custom_print(f"Summary of {col}:\n{census_df_clean[col].value_counts(dropna=False)}")


    ## Exploratory Data Analysis II
    # variables to keep
    ## get column names of new one hot encoded variables
    tempnames = []
    for string in discrete_vars:
        discrete_cols = [col for col in census_df_clean.columns if string in col]
        tempnames.extend(discrete_cols)
    tempnames = list(set(tempnames))
    ## list of categorical variable names
    vars_keep_categorical = ['new_education','race','hispanic_origin','sex','member_of_a_labor_union','region_of_previous_residence','live_in_this_house_1_year_ago','migration_prev_res_in_sunbelt','year','new_class_of_worker','moved','both_parents_present','american_father','american_mother']
    ## drop original discrete variable column names since they have been replaced by one hot encoded variables
    vars_keep_categorical = [x for x in vars_keep_categorical if x not in discrete_vars]
    ## add column names of new one hot encoded variables to list of categorical variable names
    vars_keep_categorical.extend(tempnames)
    vars_keep_categorical = list(set(vars_keep_categorical))
    ## list of numerical variable names
    vars_keep_numerical = ['age','wage_per_hour','capital_gains','capital_losses','dividends_from_stocks','weeks_worked_in_year']


    if MAKE_PLOTS:
        if SAVE_PLOTS:
            plotfilename = f'{out_dir}/EDA-{label}.pdf'
            pdf = PdfPages(plotfilename)

        # barplot of target variable
        plt.figure()
        ax = census_df_clean['target'].value_counts().plot(kind="bar")
        for container in ax.containers:
            ax.bar_label(container, label_type='edge', fmt='%.0f')
        ax.set_title("Target Variable Distribution")
        ax.set_xlabel("Income Level")
        ax.set_ylabel("Count")
        plt.tight_layout()
        if SAVE_PLOTS:
            pdf.savefig()
        plt.close()

        # histograms of numerical variables
        for col in vars_keep_numerical:
            plt.figure()
            census_df_clean[col].hist(bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            if SAVE_PLOTS:
                pdf.savefig()
            plt.close()

        # boxplots showing distribution of target variable for each numerical variable
        for col in vars_keep_numerical:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='target', y=col, data=census_df_clean, showfliers=False, palette="Set2")
            plt.title(f"Distribution of {col} by Income Level")
            plt.xlabel("Income Level")
            plt.ylabel(col)
            if SAVE_PLOTS:
                pdf.savefig()
            plt.close()

        # barplots of categorical variables
        for col in vars_keep_categorical:
            plt.figure()
            ax = census_df_clean[col].value_counts().plot(kind="bar")
            for container in ax.containers:
                ax.bar_label(container, label_type='edge', fmt='%.0f')
            ax.set_title(f"Count of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.tight_layout()
            if SAVE_PLOTS:
                pdf.savefig()
            plt.close()

        # stacked bar charts showing distribution of categorical variables by the target variable
        for col in vars_keep_categorical:
            group_counts = census_df_clean.groupby(["target", col]).size().unstack(fill_value=0)
            plt.figure()
            ax = group_counts.plot(kind="bar", stacked=True)
            for container in ax.containers:
                ax.bar_label(container, label_type='center', fmt='%.0f')
            ax.set_title(f"Distribution of '{col}' by Income Level")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.tight_layout()
            if SAVE_PLOTS:
                pdf.savefig()
            plt.close()

        if SAVE_PLOTS:
            pdf.close()
            custom_print(f"All plots saved to '{plotfilename}'")

    if FULL_VERBOSE:
        # mean of target variable levels in each numerical variable
        census_df_clean.groupby("target")[vars_keep_numerical].mean()

    if FULL_VERBOSE:
        # calculate % of males and females in each income level
        gender_counts = census_df_clean.groupby(["target", "sex"]).size().unstack(fill_value=0)
        gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
        print(gender_percentages)

        # print out the percentages
        less_than_50k_female = gender_percentages.loc['lessthan50k', 'Female']
        less_than_50k_male = gender_percentages.loc['lessthan50k', 'Male']
        more_than_50k_female = gender_percentages.loc['morethan50k', 'Female']
        more_than_50k_male = gender_percentages.loc['morethan50k', 'Male']

        print(f"{less_than_50k_female:.2f}% women in the 'less than 50k' group compared to {more_than_50k_female:.2f}% in the 'more than 50k' group")
        print(f"{less_than_50k_male:.2f}% men in the 'less than 50k' group compared to {more_than_50k_male:.2f}% in the 'more than 50k' group")

        # "both_parents_present" has only 2 observations for "morethan50k" hence dropping it
        #census_df_clean = census_df_clean.drop(columns=['both_parents_present'])
        vars_keep_categorical = [var for var in vars_keep_categorical if var != 'both_parents_present']

        # keep only the categorical and numerical variables to be used for modeling
        census_df_final = census_df_clean[vars_keep_categorical + vars_keep_numerical + ['target']]

        # print number of rows and columns in the final census dataset
        custom_print(f"Number of rows (final): {census_df_final.shape[0]}")
        custom_print(f"Number of columns (final): {census_df_final.shape[1]}")

        # show what the data looks like
        custom_print(f"Cleaned data:\n{census_df_final.head()}")

    census_df_final.to_csv(f'{out_dir}/cleaned_data-{label}.csv', index=True)

    if not trainingrun:
        return(census_df_final)

    if trainingrun:
        # setting ml models to build
        ml_models = ['rf','xgboost','lightgbm','lr','knn','dt']

        # pycaret setup
        setup(data=census_df_final,
              target='target',
              train_size=0.8,
              verbose=True,
              session_id=123,
              experiment_name="census-training",
              categorical_features=vars_keep_categorical,
              numeric_features=vars_keep_numerical,
              remove_outliers=False,
              preprocess=True,
              normalize=True,
              normalize_method='zscore',
              transformation=True,
              transformation_method='quantile',
              fix_imbalance=True,
              fix_imbalance_method='smote',
              fold=5,
              fold_strategy='stratifiedkfold',
              fold_shuffle=True,
              numeric_imputation = 'median',
              categorical_imputation = 'mode'
        )

        # show difference in outcome variable after applying SMOTE
        custom_print(f"outcome var distribution pre-SMOTE:\n{get_config('y_train').value_counts()}")
        custom_print(f"outcome var distribution post-SMOTE:\n{get_config('y_train_transformed').value_counts()}")

        # plot difference in outcome variable after applying SMOTE
        if MAKE_PLOTS:
            plt.figure()
            ax = census_df_final['target'].value_counts().plot(kind='bar', color='blue', alpha=0.5, label='Original')
            get_config('y_train_transformed').value_counts().plot(kind='bar', color='red', alpha=0.5, label='SMOTE')
            plt.legend()
            plt.title('Target Variable Distribution')
            plt.xlabel('Income Level')
            plt.ylabel('Count')
            for container in ax.containers:
                ax.bar_label(container, label_type='edge', fmt='%.0f')
            plt.tight_layout()
            if SAVE_PLOTS:
                plt.savefig(f'{out_dir}/SMOTEeffect.pdf')
            plt.close()

        # compare training set metrics between different classification models to find the best performer
        best_models = compare_models(include=ml_models, sort='AUC', n_select=7)

        # print hyperparameters for all models
        for idx in range(len(best_models)):
            print(f"Model {idx+1} Hyperparameters:")
            print(best_models[idx])
            print("#" * 80)

        # compare testing set metrics between different classification models to find the best performer
        for idx in range(len(best_models)):
            mytemp = predict_model(best_models[idx])

        # finalize the best performing model
        final_model = finalize_model(best_models[0])
        save_model(final_model, f'{out_dir}/final_model', model_only=False, verbose=True)

        return(final_model)

# define function to perform prediction
def pred_data(ml_model, test_data, out_dir):
    pred_results = predict_model(ml_model, test_data)
    pred_results.to_csv(f'{out_dir}/final_pred_results.csv', index=True)
    return(pred_results)

# define function to visualize results
def viz_results(ml_model, y_true, y_scores, out_dir):
    # visualizing the results
    from sklearn.metrics import (roc_curve, auc)
    
    # confusion matrix
    plot_model(ml_model, plot='confusion_matrix', save=out_dir)
    
    # ROC curve
    # from training and testing data
    plot_model(ml_model, plot='auc', save=out_dir)
    
    # from validation data
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label='lessthan50k')
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{out_dir}/AUC-validation.pdf', bbox_inches='tight')
    plt.close()
    
    
    # precision-recall curve
    plot_model(ml_model, plot='pr', save=out_dir)
    
    # feature importance
    plot_model(ml_model, plot='feature_all', save=out_dir)
    
    return(custom_print('Visualization complete.'))

# # prep training data and build ml model
# my_train_model = process_data(inputdata = '0_raw/census_income_learn.csv', metadata = '0_raw/census_col_labels.csv', out_dir = '1_results', label = 'train', trainingrun=True)

# # prep test data
# my_test_data = process_data(inputdata = '0_raw/census_income_test.csv', metadata = '0_raw/census_col_labels.csv', out_dir = '1_results', label = 'test', trainingrun=False)

# # run prediction on test data
# pred_results = pred_data(ml_model=my_train_model, test_data=my_test_data, out_dir='1_results/')

# if SAVE_PLOTS:
#     # visualize model training and prediction performance
#     viz_results(ml_model=my_train_model, y_true=pred_results['target'], y_scores=pred_results['prediction_score'], out_dir='1_results')