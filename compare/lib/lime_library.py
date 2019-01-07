#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 05:20:12 2018

@author: seojin.bang
"""

"""
Functions for explaining classifiers that use Image data.
"""
#from __future__ import print_function
import copy
import torch
import time
from torch.nn import functional as F
from functools import partial
import sys
sys.path.append('../')
from utils import index_transfer, cuda

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.linear_model import Ridge, lars_path, LogisticRegression
from sklearn.utils import check_random_state
from skimage.color import gray2rgb

#from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.linear_model import Ridge, lars_path, LogisticRegression
#%%

class TextExplanationModified(object):
    def __init__(self, test, segments):
        """Init function.
        Args:
            text: text
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.text = text
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None
        
class ImageExplanationModified(object):
    def __init__(self, image, segments):
        """Init function.
        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: TODO
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
                    # temp[segments == f, cp] *= 0.5
            return temp, mask
#%%
class LimeTextExplainerModified(object):

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None, is_cuda = False, dataset = None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBaseModified(kernel_fn, verbose, random_state=self.random_state)
        self.is_cuda = is_cuda
        self.dataset = dataset
        
    def explain_instance(self, text, filter_size, classifier_fn, labels=(1,),
                         hide_color=0,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         segments=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            text: text
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
           
        """
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segments is None:

            if segmentation_fn is None:
                segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            try:
                segments = segmentation_fn(text) # (1, 28, 28)

            except ValueError as e:
                raise e
        
        print(text.size())
        chunked_text = copy.deepcopy(text)
        chunked_text = chunked_text.squeeze(0)
        chunked_text = chunked_text.cpu().numpy()
        print(chunked_text.shape)
        
        if filter_size[0] * filter_size[1] > 1:
            print(segments.shape)# (1, 15, 50)
            print(chunked_text.shape) # 750 * 100
            print(np.mean(chunked_text[segments ==1]))
            for x in np.unique(segments):
                chunked_text[segments == x] = np.mean(chunked_text[segments == x])
   
        fudged_text = copy.deepcopy(chunked_text)
        fudged_text[:] = hide_color
        
        top = labels

        #print(torch.Tensor(chunked_text).unsqueeze(0).size())
        data, labels, neighborhood_data = self.data_labels(#text,
            torch.Tensor(chunked_text).unsqueeze(0),
                                        fudged_text, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)

        if filter_size[0] * filter_size[1] > 1:
            neighborhood_data = F.avg_pool2d(torch.Tensor(neighborhood_data).view(num_samples, segments.shape[-2], segments.shape[-1]), kernel_size = filter_size, stride = filter_size, padding = 0)
            neighborhood_data = neighborhood_data.cpu().numpy().reshape(num_samples, neighborhood_data.size(-2) * neighborhood_data.size(-1))
            
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = TextExplanationModified(chunked_text, segments)
       
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        t3 = time.time()
        for label in ret_exp.top_labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp_local_pred_proba, ret_exp_local_pred) = self.base.explain_instance_with_data(
                neighborhood_data, labels,
                distances, label, num_features,
                model_regressor = model_regressor,
                feature_selection = self.feature_selection)
                
            if label == ret_exp.top_labels[0]:
                ret_exp.local_pred_proba = [ret_exp_local_pred_proba]
                ret_exp.local_pred = [ret_exp_local_pred]
            else:
                #print(ret_exp_score, ret_exp_local_pred)
                ret_exp.local_pred_proba.append(ret_exp_local_pred_proba)
                ret_exp.local_pred.append(ret_exp_local_pred)

        t4 = time.time()
        ret_exp.local_pred_proba = np.array(ret_exp.local_pred_proba)[np.argsort(ret_exp.top_labels)]

        return ret_exp
    
    def data_labels(self,
                    text,
                    fudged_text,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):

        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        #labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(text.squeeze(0)).cpu().numpy()
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_text[mask]
            #print(temp.shape)
            imgs.append(temp)
        
        preds = classifier_fn(cuda(torch.Tensor(imgs), self.is_cuda)).detach().cpu().numpy()
        if self.dataset == 'mnist':
            neighborhood_data = np.reshape(np.squeeze(np.stack(imgs, axis=0), axis=1), (num_samples, -1))
        elif self.dataset == 'imdb':
            neighborhood_data = np.mean(np.stack(imgs, axis=0), axis = -1)
        else:
            raise ValueError('unknown dataset')
            
        return data, np.array(preds), neighborhood_data
   

#%%
class LimeImageExplainerModified(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None, is_cuda = False, dataset = None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBaseModified(kernel_fn, verbose, random_state=self.random_state)
        self.is_cuda = is_cuda
        self.dataset = dataset
        
    def explain_instance(self, image, filter_size, classifier_fn, labels=(1,),
                         hide_color=0,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         segments=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
           
        """
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segments is None:

            if segmentation_fn is None:
                segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            try:
                segments = segmentation_fn(image) # (1, 28, 28)

            except ValueError as e:
                raise e

        #t0 = time.time()
        chunked_image = copy.deepcopy(image)
        chunked_image = chunked_image.squeeze(0)
        chunked_image = chunked_image.cpu().numpy()

        if filter_size[0] * filter_size[1] > 1:
            #image = F.avg_pool2d(image, kernel_size = filter_size, stride = filter_size, padding = 0)
            #image = F.max_unpool2d(image, kernel_size = filter_size, stride = filter_size, padding = 0)
            for x in np.unique(segments):
                chunked_image[segments == x] = np.mean(chunked_image[segments == x])
            #chunked_image = torch.Tensor(chunked_image).unsqueeze(0)
        t1 = time.time()
        #print(t1 - t0)
           
        fudged_image = copy.deepcopy(chunked_image)
        #fudged_image = fudged_image.squeeze(0)
        #fudged_image = fudged_image.numpy()
        fudged_image[:] = hide_color
        
#        if len(image.shape) == 4: 
#            image = gray2rgb(image.numpy().astype(int))
#            
#        fudged_image = image.copy() # (1, 1, 28, 28, 3)
#        if hide_color is None:
#            for x in np.unique(segments):
#                fudged_image[segments == x] = (
#                    np.mean(image[segments == x][:, 0]),
#                    np.mean(image[segments == x][:, 1]),
#                    np.mean(image[segments == x][:, 2]))
#        else:
#            fudged_image[:] = hide_color

        top = labels

        #print(torch.Tensor(chunked_image).unsqueeze(0).size())
        data, labels, neighborhood_data = self.data_labels(#image,
            torch.Tensor(chunked_image).unsqueeze(0),
                                        fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)

        if filter_size[0] * filter_size[1] > 1:
            neighborhood_data = F.avg_pool2d(torch.Tensor(neighborhood_data).view(num_samples, segments.shape[-2], segments.shape[-1]), kernel_size = filter_size, stride = filter_size, padding = 0)
            neighborhood_data = neighborhood_data.cpu().numpy().reshape(num_samples, neighborhood_data.size(-2) * neighborhood_data.size(-1))
            
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanationModified(chunked_image, segments)
       
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        t3 = time.time()
        for label in ret_exp.top_labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp_local_pred_proba, ret_exp_local_pred) = self.base.explain_instance_with_data(
                neighborhood_data, labels,
                distances, label, num_features,
                model_regressor = model_regressor,
                feature_selection = self.feature_selection)
                
            if label == ret_exp.top_labels[0]:
                ret_exp.local_pred_proba = [ret_exp_local_pred_proba]
                ret_exp.local_pred = [ret_exp_local_pred]
            else:
                #print(ret_exp_score, ret_exp_local_pred)
                ret_exp.local_pred_proba.append(ret_exp_local_pred_proba)
                ret_exp.local_pred.append(ret_exp_local_pred)

        t4 = time.time()
        ret_exp.local_pred_proba = np.array(ret_exp.local_pred_proba)[np.argsort(ret_exp.top_labels)]

        return ret_exp
    
    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        #print(segments.shape)
        #print(segments)
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        #labels = []
        data[0, :] = 1
        imgs = []
        #t0 = time.time()
        for row in data:
            temp = copy.deepcopy(image.squeeze(0)).cpu().numpy()
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            #print(temp.shape)
            imgs.append(temp)
            #print(time.time() - t0)
            #if len(imgs) == batch_size:
            #    #preds = classifier_fn(torch.Tensor(imgs)).argmax(dim = 1).detach().numpy()
            #    preds = classifier_fn(torch.Tensor(imgs)).detach().numpy()
            #    labels.extend(preds)
            #    imgs = []
        
        #preds = classifier_fn(torch.Tensor(imgs)).argmax(dim = 1).detach().numpy()
        preds = classifier_fn(cuda(torch.Tensor(imgs), self.is_cuda)).detach().cpu().numpy()
        if self.dataset == 'mnist':
            neighborhood_data = np.reshape(np.squeeze(np.stack(imgs, axis=0), axis=1), (num_samples, -1))
        elif self.dataset == 'imdb':
            neighborhood_data = np.mean(np.stack(imgs, axis=0), axis = -1)
        else:
            raise ValueError('unknown dataset')

        #if len(imgs) > 0:
        #    preds = classifier_fn(torch.Tensor(imgs)).argmax(dim = 1).detach().numpy()
        #    labels.extend([preds])
            
        return data, np.array(preds), neighborhood_data
   
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
            
            if model_regressor.multi_class in ['multinomial', 'ovr']:
                feature_weights = sorted(zip(range(data.shape[0]), clf.coef_[0] * data[0]),
                                         key=lambda x: np.abs(x[1]),
                                         reverse=True)
            else:
                feature_weights = sorted(zip(range(data.shape[0]), clf.coef_ * data[0]),
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
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
            
        elif 'LogisticRegression' in str(type(model_regressor)):
            
            if model_regressor.multi_class is 'multinomial':
                #print("Explainer: logistic regression with multinomial dist.")
                labels_column = neighborhood_labels.argmax(axis = -1)
                
            elif model_regressor.multi_class is 'ovr':
                #print("Explainer: logistic regression with binomial dist.")
                #labels_column = neighborhood_labels.argmax(axis = -1)
                labels_column = neighborhood_labels.argmax(axis = -1)
                labels_column = 1 * (labels_column == label)
                    
                if len(np.unique(labels_column)) == 1:
                    if np.sum(labels_column) == 0:
                        return 'none', 'none', 0.0, 0
                    else:
                        return 'none', 'none', 1.0, 1 
            else:
                raise KeyError('unknown model_regressor.multi_class')
                
        else:
            raise KeyError('unknown model_regressor')
        
        used_features = self.feature_selection(neighborhood_data,
                                       labels_column,
                                       weights,
                                       num_features,
                                       feature_selection, model_regressor)  
                
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
        
        #prediction_score = easy_model.score(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))[0]
        local_pred_proba = easy_model.predict_proba(neighborhood_data[0, used_features].reshape(1, -1))[0][-1]
        
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0], label)
        
        #print(local_pred_proba)
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_[0]),
                       key=lambda x: np.abs(x[1]), reverse=True),
                local_pred_proba, local_pred)    

