import numpy as np
import skimage

caffe_root = 'PATH_TO_CAFFE'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


class NetWrap(caffe.Net):
    """
    NetWrap extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean_file: path to the binaryproto file
    input_scale: not implemented
    raw_scale: typically 255
    channel_swap: should be BGR (2, 1, 0)
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean_file=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]

        if mean_file is not None:
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open( mean_file , 'rb' ).read()
            blob.ParseFromString(data)
            # Transform from K x H x W to H x W x K
            self.mean_array = np.array(caffe.io.blobproto_to_array(blob))[0].transpose((1, 2, 0))

        if input_scale is not None:
            self.input_scale = input_scale
        if raw_scale is not None:
            self.raw_scale = raw_scale
        if channel_swap is not None:
            self.channel_swap = channel_swap

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        """
        Predict classification probabilities of inputs.
        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.
        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            #input_[ix] = caffe.io.resize_image(in_, self.image_dims)
            input_[ix] = (skimage.transform.resize(in_, self.image_dims) * self.raw_scale)\
                            [:, :, self.channel_swap] - self.mean_array

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]


        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = in_.transpose((2, 0, 1))
            # caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = self.blobs["prob"].data

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions

    def feature_extraction(self, inputs, layer):

        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            #input_[ix] = caffe.io.resize_image(in_, self.image_dims)
            input_[ix] = (skimage.transform.resize(in_, self.image_dims) * self.raw_scale)\
                            [:, :, self.channel_swap] - self.mean_array

        # Take center crop.
        center = np.array(self.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -self.crop_dims / 2.0,
            self.crop_dims / 2.0
        ])
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = in_.transpose((2, 0, 1))
            # caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})

        return self.blobs[layer].data

MODEL_PREFIX = "PATH_TO_MODEL_FOLDER"
WEIGHT_FILE = MODEL_PREFIX + "siamese_vgg_food_train_iter_100000.caffemodel"
DEPLOY_FILE = MODEL_PREFIX + "deploy.prototxt"
MEAN_FILE = MODEL_PREFIX + "food_mean.binaryproto"

net_feature = NetWrap(model_file=DEPLOY_FILE,
                pretrained_file=WEIGHT_FILE,
                image_dims=(256, 256),
                mean_file=MEAN_FILE,
                raw_scale=255,
                channel_swap=(2, 1, 0))

# Sample for feature extraction
# feature = net_feature.feature_extraction([image_data], "feature")[0].copy()
