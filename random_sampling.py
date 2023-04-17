import numpy as np
import os
import random
from formate_data_classes import build_class_dictionaries

def determine_num_crops(depressed_dict, normal_dict, crop_width=125):
    merged_dict = {**normal_dict, **depressed_dict}
    shortest_clip = min(merged_dict.items(), key=lambda x: x[1].shape[1])
    shortest_pixel_width = shortest_clip[1].shape[1]
    num_samples_from_clips = shortest_pixel_width / crop_width
    return num_samples_from_clips


def build_class_sample_dict(segmented_audio_dict, n_samples, crop_width):
    class_samples_dict = dict()
    for partic_id, clip_mat in segmented_audio_dict.items():
            samples = get_random_samples(clip_mat, n_samples, crop_width)
            class_samples_dict[partic_id] = samples
    return class_samples_dict


def get_random_samples(matrix, n_samples, crop_width):
    # crop full spectrogram into segments of width = crop_width
    clipped_mat = matrix[:, (matrix.shape[1] % crop_width):]
    n_splits = clipped_mat.shape[1] / crop_width
    cropped_sample_ls = np.split(clipped_mat, n_splits, axis=1)

    print(cropped_sample_ls)
    # get random samples
    samples = random.sample(cropped_sample_ls, 20)
    return samples


def create_sample_dicts(crop_width):
   
    depressed_dict, normal_dict = build_class_dictionaries(r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/processed')
    n_samples = determine_num_crops(depressed_dict, normal_dict,
                                    crop_width=crop_width)
    print("n_sampe")
    print(n_samples)
    depressed_samples = build_class_sample_dict(depressed_dict, n_samples,
                                                crop_width)
    print("de sample")
    print(depressed_samples)
    
    normal_samples = build_class_sample_dict(normal_dict, n_samples,
                                             crop_width)
    print("no sample")
    print(normal_samples)
    for key, _ in depressed_samples.items():
        path = r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/processed'
        filename = 'D{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *depressed_samples[key])
    # save normal arrays to .npz
    for key, _ in normal_samples.items():
        path = r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/processed'
        filename = '/N{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *normal_samples[key])

def rand_samp_train_test_split(npz_file_dir):

    npz_files = os.listdir(npz_file_dir)

    dep_samps = [f for f in npz_files if f.startswith('D')]
    norm_samps = [f for f in npz_files if f.startswith('N')]
    max_samples = min(len(dep_samps), len(norm_samps))

    dep_select_samps = np.random.choice(dep_samps, size=max_samples,
                                        replace=False)
    norm_select_samps = np.random.choice(norm_samps, size=max_samples,
                                         replace=False)

    test_size = 0.2
    num_test_samples = int(len(dep_select_samps) * test_size)

    train_samples = []
    for sample in dep_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    for sample in norm_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    train_labels = np.concatenate((np.ones(int(len(train_samples)/2)),
                                   np.zeros(int(len(train_samples)/2))))

    test_samples = []
    for sample in dep_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    for sample in norm_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    test_labels = np.concatenate((np.ones(int(len(test_samples)/2)),
                                  np.zeros(int(len(test_samples)/2))))

    return np.array(train_samples), train_labels, np.array(test_samples), \
        test_labels


if __name__ == '__main__':
    dir_name = r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/processed'
    create_sample_dicts(crop_width=300)

    train_samples, train_labels, test_samples, \
        test_labels = rand_samp_train_test_split(dir_name)

    # save as npz locally
    print("Saving npz file locally...")
    np.savez(dir_name+'/train_samples.npz', train_samples)
    np.savez(dir_name+'/train_labels.npz', train_labels)
    np.savez(dir_name+'/test_samples.npz', test_samples)
    np.savez(dir_name+'/test_labels.npz', test_labels)