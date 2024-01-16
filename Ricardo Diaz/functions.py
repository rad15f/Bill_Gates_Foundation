import pandas as pd
import numpy as np


def common_var_checker(df_train, df_val, df_test, target):
    """
    The common variables checker

    Parameters
    ----------
    df_train : the dataframe of training data
    df_val : the dataframe of validation data
    df_test : the dataframe of test data
    target : the name of the target

    Returns
    ----------
    The dataframe of common variables between the training, validation and test data
    """

    # Get the dataframe of common variables between the training, validation and test data
    df_common_var = pd.DataFrame(
        np.intersect1d(np.intersect1d(df_train.columns, df_val.columns), np.union1d(df_test.columns, [target])),
        columns=['common var'])

    return df_common_var


def id_checker(df, dtype='float'):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    dtype : the data type identifiers cannot have, 'float' by default
            i.e., if a feature has this data type, it cannot be an identifier

    Returns
    ----------
    The dataframe of identifiers
    """

    # Get the dataframe of identifiers
    df_id = df[[var for var in df.columns
                # If the data type is not dtype
                if (df[var].dtype != dtype
                    # If the value is unique for each sample
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]

    return df_id


def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : the dataframe

    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and data type
    """

    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan


def cat_var_checker(df, dtype='object'):
    """
    The categorical variable checker

    Parameters
    ----------
    df : the dataframe
    dtype : the data type categorical variables should have, 'object' by default
            i.e., if a variable has this data type, it should be a categorical variable

    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """

    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           # If the data type is dtype
                           for var in df.columns if df[var].dtype == dtype],
                          columns=['var', 'nunique'])

    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat


def separate_duplicate_original(X_aug_train, y_aug_train, minor_class):
    """
    Separate the duplicated class from the original class

    Parameters
    ----------
    X_aug_train : The augmented feature matrix
    y_aug_train : The augmented target vector
    minor_class : The minority class

    Returns
    ----------
    The separated duplicated class and original class
    """

    # Make a copy of y_aug_train
    y_aug_dup_ori_train = np.array(y_aug_train)

    # For each sample in the augmented data
    for i in range(X_aug_train.shape[0]):
        # If the sample has the minor class
        if y_aug_dup_ori_train[i] == minor_class:
            # Flag variable, indicating whether a sample in the augmented data is the same as a sample in the original data
            same = False

            # For each sample in the original data
            for j in range(X_aug_train.shape[0]):
                if j == i:
                    continue

                # If the sample has the minor class
                if y_aug_dup_ori_train[j] == minor_class:
                    if len(np.setdiff1d(X_aug_train[i, :], X_aug_train[j, :])) == 0:
                        # The two samples are the same
                        same = True
                        break

            # If the two samples are different
            if same is False:
                y_aug_dup_ori_train[i] = 2

    return y_aug_dup_ori_train


def separate_generate_original(X_aug_train, y_aug_train, X_train, y_train, minor_class):
    """
    Separate the generated class from the original class

    Parameters
    ----------
    X_aug_train : The augmented feature matrix
    y_aug_train : The augmented target vector
    X_train : The original feature matrix
    y_train : The original target vector
    minor_class : The minority class

    Returns
    ----------
    The separated generated class and original class
    """

    # Make a copy of y_aug_train
    y_aug_gen_ori_train = np.array(y_aug_train)

    # For each sample in the augmented data
    for i in range(X_aug_train.shape[0]):
        # If the sample has the minor class
        if y_aug_gen_ori_train[i] == minor_class:
            # Flag variable, indicating whether a sample in the augmented data is the same as a sample in the original data
            same = False

            # For each sample in the original data
            for j in range(X_train.shape[0]):
                # If the sample has the minor class
                if y_train[j] == minor_class:
                    if len(np.setdiff1d(X_aug_train[i, :], X_train[j, :])) == 0:
                        # The two samples are the same
                        same = True
                        break

            # If the two samples are different
            if same is False:
                y_aug_gen_ori_train[i] = 2

    return y_aug_gen_ori_train


from sklearn.model_selection import PredefinedSplit


def get_train_val_ps(X_train, y_train, X_val, y_val):
    """
    Get the:
    feature matrix and target velctor in the combined training and validation data
    target vector in the combined training and validation data
    PredefinedSplit

    Parameters
    ----------
    X_train : the feature matrix in the training data
    y_train : the target vector in the training data
    X_val : the feature matrix in the validation data
    y_val : the target vector in the validation data

    Return
    ----------
    The feature matrix in the combined training and validation data
    The target vector in the combined training and validation data
    PredefinedSplit
    """

    # Combine the feature matrix in the training and validation data
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector in the training and validation data
    y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).reshape(-1)

    # Get the indices of training and validation data
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps