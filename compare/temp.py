#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:39:42 2018

@author: seojin.bang
"""

from __future__ import print_function
import numpy as np
from sklearn.linear_model import Ridge, lars_path, LogisticRegression
from sklearn.utils import check_random_state



#%%
class LimeBaseModified(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function
        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.
        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel
        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features, model_regressor = None):
        """Iteratively adds features to the model"""
        
        if model_regressor is None:
            clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        else:
            clf = model_regressor
            
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method, model_regressor):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features, model_regressor)
        elif method == 'highest_weights':
            if model_regressor is None:
                clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
            else:
                clf = model_regressor
                
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method, model_regressor)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.
        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()
        Returns:
            (intercept, exp, score):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
        """

        weights = self.kernel_fn(distances)

        if model_regressor is None:
            labels_column = neighborhood_labels[:, label]
            used_features = self.feature_selection(neighborhood_data,
                                                   labels_column,
                                                   weights,
                                                   num_features,
                                                   feature_selection, 
                                                   model_regressor)
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        else:
            labels_column = neighborhood_labels.argmax(axis = -1)
            used_features = self.feature_selection(neighborhood_data,
                                                   labels_column,
                                                   weights,
                                                   num_features,
                                                   feature_selection, 
                                                   model_regressor)    
            
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
        
        prediction_score = easy_model.score(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0], label)
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
                
#%%
kernel_width=.25
kernel=None
verbose=False
feature_selection='auto'
random_state=None
kernel_width = float(kernel_width)

if kernel is None:
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

kernel_fn = partial(kernel, kernel_width=kernel_width)
        
neighborhood_data = data
neighborhood_labels = labels
#model_regressor = None
feature_selection = 'auto'
self = LimeBase(kernel_fn, verbose, random_state=None)
#self = lime_base.LimeBase(kernel_fn, verbose, random_state=None)
#self.base.explain_instance_with_data(
#                data, labels, distances, label, num_features,
#                model_regressor=model_regressor,
#                feature_selection=self.feature_selection)


#distances = distances
#label = label
#num_features = num_features

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
y_vec = mnist.target.astype(np.uint8)
#%%
#%matplotlib inline
fig, ax1 = plt.subplots(1,1)
ax1.imshow(X_vec[0], interpolation = 'none')
ax1.set_title('Digit: {}'.format(y_vec[0]))
#%%
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)


makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_rf_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    #('Normalize', Normalizer()),
    #('PCA', PCA(16)),
    ('RF', RandomForestClassifier())
                              ])
    
    #%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.55)   
simple_rf_pipeline.fit(X_train, y_train)

#%%
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
    
#%%
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
#%%
explanation = explainer.explain_instance(X_test[0], 
                                         classifier_fn = simple_rf_pipeline.predict_proba, 
                                         top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
#%%
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
ax1.set_title('Positive Regions for {}'.format(y_test[0]))
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(y_test[0]))



#%%
image = X_test[0]
classifier_fn = simple_rf_pipeline.predict_proba
top_labels=10
hide_color=0
num_samples=10000
segmentation_fn=segmenter
labels=(1,)

batch_size=10
distance_metric='cosine'
model_regressor=None
random_seed=None














