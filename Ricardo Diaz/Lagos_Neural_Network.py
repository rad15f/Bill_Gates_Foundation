import tensorflow as tf
import numpy as np
# Get the absolute path of the current folder
abspath_curr = '/home/ubuntu/Lagos'
city = 'Lagos'
import pandas as pd
# Get the Dataset
df_train = pd.read_csv(abspath_curr + '/dataset/100m/All_Features_Lagos_Train.csv',
                           header=0).loc[:792,:]

df_val = pd.read_csv(abspath_curr + '/dataset/100m/All_Features_Lagos_Train.csv',
                          header=0).iloc[792:,:]
# Select the Columns you want
best_features = ['uu_bld_count_2020',
 'ph_dist_inland_water_2018',
 'ses_child_stuned_2014',
 'ses_m_lit_2014',
 'uu_bld_den_2020',
 'gabor_sc7_filter_11',
 'lbpm_sc7_mean',
 'ph_dist_open_coast_2020',
 'ndvi_sc5_variance',
 'ndvi_sc7_variance',
 'ph_land_c2_2020',
 'gabor_sc3_filter_10',
 'gabor_sc3_filter_11',
 'ph_dist_cultivated_2015',
 'in_night_light_2016',
 'ph_dist_shrub_2015',
 'fourier_sc71_mean',
 'lbpm_sc5_mean',
 'sfs_sc71_std',
 'ph_ndvi_2019',
 'n',
 'po_hrsl_2018',
 'ses_unmet_need_2014',
 'ses_pfpr_2017',
 'in_dist_waterway_2016',
 'r',
 'ph_dist_aq_veg_2015',
 'ph_dist_art_surface_2015',
 'uu_urb_bldg_2018',
 'ph_pm25_2016',
 'ph_base_water_2010',
 'sfs_sc31_max_line_length',
 'ses_preg_2017',
 'ph_dist_riv_network_2007',
 'pantex_sc3_min',
 'b',
 'hog_sc7_kurtosis',
 'hog_sc3_mean',
 'fs_dist_school_2020',
 'hog_sc7_skew',
 'fs_dist_fs_2020',
 'sh_ethno_den_2020',
 'hog_sc7_max',
 'po_wp_2020',
 'orb_sc51_max',
 'fourier_sc31_variance',
 'lbpm_sc3_mean',
 'sfs_sc31_std',
 'sfs_sc51_mean',
 'lbpm_sc3_kurtosis',
 'ph_grd_water_2000',
 'fourier_sc71_variance',
 'lsr_sc31_line_mean',
 'sfs_sc51_std',
 'hog_sc3_skew',
'labels']
contextual_features = ['fourier_sc31_mean',
                       'fourier_sc31_variance',
                       'fourier_sc51_mean',
                       'fourier_sc51_variance',
                       'fourier_sc71_mean',
                       'fourier_sc71_variance',
                       'gabor_sc3_filter_1',
                       'gabor_sc3_filter_10',
                       'gabor_sc3_filter_11',
                       'gabor_sc3_filter_12',
                       'gabor_sc3_filter_13',
                       'gabor_sc3_filter_14',
                       'gabor_sc3_filter_2',
                       'gabor_sc3_filter_3',
                       'gabor_sc3_filter_4',
                       'gabor_sc3_filter_5',
                       'gabor_sc3_filter_6',
                       'gabor_sc3_filter_7',
                       'gabor_sc3_filter_8',
                       'gabor_sc3_filter_9',
                       'gabor_sc3_mean',
                       'gabor_sc3_variance',
                       'gabor_sc5_filter_1',
                       'gabor_sc5_filter_10',
                       'gabor_sc5_filter_11',
                       'gabor_sc5_filter_12',
                       'gabor_sc5_filter_13',
                       'gabor_sc5_filter_14',
                       'gabor_sc5_filter_2',
                       'gabor_sc5_filter_3',
                       'gabor_sc5_filter_4',
                       'gabor_sc5_filter_5',
                       'gabor_sc5_filter_6',
                       'gabor_sc5_filter_7',
                       'gabor_sc5_filter_8',
                       'gabor_sc5_filter_9',
                       'gabor_sc5_mean',
                       'gabor_sc5_variance',
                       'gabor_sc7_filter_1',
                       'gabor_sc7_filter_10',
                       'gabor_sc7_filter_11',
                       'gabor_sc7_filter_12',
                       'gabor_sc7_filter_13',
                       'gabor_sc7_filter_14',
                       'gabor_sc7_filter_2',
                       'gabor_sc7_filter_3',
                       'gabor_sc7_filter_4',
                       'gabor_sc7_filter_5',
                       'gabor_sc7_filter_6',
                       'gabor_sc7_filter_7',
                       'gabor_sc7_filter_8',
                       'gabor_sc7_filter_9',
                       'gabor_sc7_mean',
                       'gabor_sc7_variance',
                       'hog_sc3_kurtosis',
                       'hog_sc3_max',
                       'hog_sc3_mean',
                       'hog_sc3_skew',
                       'hog_sc3_variance',
                       'hog_sc5_kurtosis',
                       'hog_sc5_max',
                       'hog_sc5_mean',
                       'hog_sc5_skew',
                       'hog_sc5_variance',
                       'hog_sc7_kurtosis',
                       'hog_sc7_max',
                       'hog_sc7_mean',
                       'hog_sc7_skew',
                       'hog_sc7_variance',
                       'lac_sc3_lac',
                       'lac_sc5_lac',
                       'lac_sc7_lac',
                       'lbpm_sc3_kurtosis',
                       'lbpm_sc3_max',
                       'lbpm_sc3_mean',
                       'lbpm_sc3_skew',
                       'lbpm_sc3_variance',
                       'lbpm_sc5_kurtosis',
                       'lbpm_sc5_max',
                       'lbpm_sc5_mean',
                       'lbpm_sc5_skew',
                       'lbpm_sc5_variance',
                       'lbpm_sc7_kurtosis',
                       'lbpm_sc7_max',
                       'lbpm_sc7_mean',
                       'lbpm_sc7_skew',
                       'lbpm_sc7_variance',
                       'lsr_sc31_line_contrast',
                       'lsr_sc31_line_length',
                       'lsr_sc31_line_mean',
                       'lsr_sc51_line_contrast',
                       'lsr_sc51_line_length',
                       'lsr_sc51_line_mean',
                       'lsr_sc71_line_contrast',
                       'lsr_sc71_line_length',
                       'lsr_sc71_line_mean',
                       'mean_sc3_mean',
                       'mean_sc3_variance',
                       'mean_sc5_mean',
                       'mean_sc5_variance',
                       'mean_sc7_mean',
                       'mean_sc7_variance',
                       'ndvi_sc3_mean',
                       'ndvi_sc3_variance',
                       'ndvi_sc5_mean',
                       'ndvi_sc5_variance',
                       'ndvi_sc7_mean',
                       'ndvi_sc7_variance',
                       'orb_sc31_kurtosis',
                       'orb_sc31_max',
                       'orb_sc31_mean',
                       'orb_sc31_skew',
                       'orb_sc31_variance',
                       'orb_sc51_kurtosis',
                       'orb_sc51_max',
                       'orb_sc51_mean',
                       'orb_sc51_skew',
                       'orb_sc51_variance',
                       'orb_sc71_kurtosis',
                       'orb_sc71_max',
                       'orb_sc71_mean',
                       'orb_sc71_skew',
                       'orb_sc71_variance',
                       'pantex_sc3_min',
                       'pantex_sc5_min',
                       'pantex_sc7_min',
                       'sfs_sc31_max_line_length',
                       'sfs_sc31_max_ratio_of_orthogonal_angles',
                       'sfs_sc31_mean',
                       'sfs_sc31_min_line_length',
                       'sfs_sc31_std',
                       'sfs_sc31_w_mean',
                       'sfs_sc51_max_line_length',
                       'sfs_sc51_max_ratio_of_orthogonal_angles',
                       'sfs_sc51_mean',
                       'sfs_sc51_min_line_length',
                       'sfs_sc51_std',
                       'sfs_sc51_w_mean',
                       'sfs_sc71_max_line_length',
                       'sfs_sc71_max_ratio_of_orthogonal_angles',
                       'sfs_sc71_mean',
                       'sfs_sc71_min_line_length',
                       'sfs_sc71_std',
                       'sfs_sc71_w_mean']
covariate_features = ['fs_dist_fs_2020',
                      'fs_dist_school_2020',
                      'in_dist_rd_2016',
                      'in_dist_rd_intersect_2016',
                      'in_dist_waterway_2016',
                      'in_night_light_2016',
                      'ph_base_water_2010',
                      'ph_bio_dvst_2015',
                      'ph_climate_risk_2020',
                      'ph_dist_aq_veg_2015',
                      'ph_dist_art_surface_2015',
                      'ph_dist_bare_2015',
                      'ph_dist_cultivated_2015',
                      'ph_dist_herb_2015',
                      'ph_dist_inland_water_2018',
                      'ph_dist_open_coast_2020',
                      'ph_dist_shrub_2015',
                      'ph_dist_sparse_veg_2015',
                      'ph_dist_woody_tree_2015',
                      'ph_gdmhz_2005',
                      'ph_grd_water_2000',
                      'ph_hzd_index_2011',
                      'ph_land_c1_2019',
                      'ph_land_c2_2020',
                      'ph_max_tem_2019',
                      'ph_ndvi_2019',
                      'ph_pm25_2016',
                      'ph_slope_2000',
                      'ses_an_visits_2014',
                      'ses_child_stuned_2014',
                      'ses_dtp3_2014',
                      'ses_hf_delivery_2014',
                      'ses_impr_water_src_2014',
                      'ses_ITN_2014',
                      'ses_m_lit_2014',
                      'ses_measles_2014',
                      'ses_odef_2014',
                      'ses_pfpr_2017',
                      'ses_preg_2017',
                      'ses_unmet_need_2014',
                      'ses_w_lit_2016',
                      'sh_dist_mnr_pofw_2019',
                      'sh_dist_pofw_2019',
                      'sh_ethno_den_2020',
                      'uu_bld_count_2020',
                      'uu_bld_den_2020',
                      'ho_impr_housing_2015',
                      'fs_dist_hf_2019',
                      'po_hrsl_2018',
                      'po_wp_2020',
                      'ph_dist_riv_network_2007',
                      'uu_urb_bldg_2018',
                      'ses_dist_gov_office_2022']
rgbn_features = ['r', 'g', 'b', 'n','labels']
columns = contextual_features + covariate_features + rgbn_features
df_train = df_train[columns]
df_val = df_val[columns]
target = 'labels'
# Get the feature matrix
train_x = df_train[np.setdiff1d(df_train.columns, [target])].values
test_x = df_val[np.setdiff1d(df_val.columns, [target])].values
# Get the target vector
train_y = df_train[target].values
test_y = df_val[target].values
'''
Keras
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adagrad
from sklearn.metrics import f1_score
# Generate some example data
num_samples = 793
input_size = 201
# Create a more complex MLP model in Keras for binary classification
model = Sequential()
model.add(Dense(256, input_dim=input_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adagrad optimizer
optimizer = Adagrad(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with a batch size of 32 and 50 epochs, with early stopping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(train_x, train_y, batch_size=32, epochs=50, validation_data=(test_x, test_y), callbacks=[early_stopping])


from sklearn.metrics import classification_report

# Evaluate the model on the test set and generate a classification report
y_pred = model.predict(test_x)
y_pred = np.round(y_pred).astype(int)
report = classification_report(test_y, y_pred)

print(report)

from keras.models import save_model
# Save the trained model to a file
save_model(model, 'my_model.h5')

from keras.models import load_model
# Load the saved model from a file
loaded_model = load_model('my_model.h5')

'''
In this code, we first define a more complex MLP model for binary classification, with three hidden layers and varying numbers of units and activation functions. We also add dropout layers after each hidden layer to prevent overfitting.

We then compile the model using Adagrad optimizer with a learning rate of 0.01, which adapts the learning rate for each parameter based on its historical gradient information.

During training, we use early stopping with a patience of 5 epochs, which stops the training process early if the validation loss does not improve for 5 consecutive epochs. This helps prevent overfitting and ensures that the model is not trained for too long.'''