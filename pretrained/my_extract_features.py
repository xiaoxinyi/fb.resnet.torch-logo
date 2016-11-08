#!/usr/local/bin/ipython -i
import PyTorch
import PyTorchHelpers

model_path_default = 'models-torch/resnet-18.t7'
def extract_featrues(image_dir, list_of_filenames=False, model_path=model_path_default, batch_size=1):
    """
    This method provides functionality of extract features for images given a pretrained model.

    @params:
    model_path : path to the trained model e.g. models-torch/resnet-18.t7
    image_dir : directory of images to be predicted e.g. ../data/paralogo
    list_of_filenames : e.g. ['dir/1.jpg', 'dir/2.jpg'] Note: False in python (not None)= nil in lua
    batch_size : #image test for one epoch
    @return :
    features : a float tensor for image features
    """

    # Load a lua class from a lua file
    FeatureExtractor = PyTorchHelpers.load_lua_class('my-extract-features.lua', 'FeatureExtractor')

    # Construct a object for the class
    fe = FeatureExtractor(model_path)

    # Feature extraction mothod called
    features = fe.extract(image_dir, list_of_filenames, batch_size)

    return features


if __name__ == '__main__':
    # model_path = 'models-torch/resnet-18.t7'
    image_dir = 'data/pradalogo'
    features = extract_featrues(image_dir)
