import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
from PIL import Image

base_dir = "../../data/DRIVE/"
data_dir = base_dir + 'converted_data/'

def convert_h5():
    train_path = base_dir+'training/'
    test_path = base_dir+'test/'
    
    train_label_list = os.listdir(train_path + '1st_manual/')
    train_label_list = [os.path.splitext(label_name)[0] for label_name in train_label_list]
    train_label_list.sort()

    train_image_list = os.listdir(train_path + 'images/')
    train_image_list = [os.path.splitext(image_name)[0] for image_name in train_image_list]
    train_image_list.sort()

    test_label_list = os.listdir(test_path + '1st_manual/')
    test_label_list = [os.path.splitext(label_name)[0] for label_name in test_label_list]
    test_label_list.sort()

    test_image_list = os.listdir(test_path + 'images/')
    test_image_list = [os.path.splitext(image_name)[0] for image_name in test_image_list]
    test_image_list.sort()

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    os.makedirs(data_dir)
    _prepare_data("train", train_path, train_image_list, train_label_list)
    _prepare_data("test", test_path, test_image_list, test_label_list)

def _prepare_data(split, path, image_list, label_list):

    _create_list_file(split, image_list)

    for image_name, label_name in zip(image_list, label_list):
        image_path = path + 'images/' + image_name + ".tif"
        
        tiff_image = Image.open(image_path)
        tiff_array = np.array(tiff_image)                
        
        label_path = path + '1st_manual/' + label_name + ".gif"
        gif_image = Image.open(label_path)
        gif_array = np.array(gif_image)

        # Create a new HDF5 file
        hdf5_path = data_dir + image_name + '.h5'
        hdf5_file = h5py.File(hdf5_path, 'w')

        # Create datasets in the HDF5 file and write the image arrays
        hdf5_file.create_dataset('image', data=tiff_array)
        hdf5_file.create_dataset('label', data=gif_array)

        # Close the HDF5 file
        hdf5_file.close()

def _create_list_file(split, image_list):
    output_file = base_dir + split + '.list'   
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            for image_name in image_list:
                f.write(image_name + '\n')

if __name__ == '__main__':
    convert_h5()