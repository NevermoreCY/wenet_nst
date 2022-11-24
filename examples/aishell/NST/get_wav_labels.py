import os
import random
import tarfile
import argparse

# python split_data_list_nst.py --job_nums=50 --data_list_path=data/train/pseudo_data_iter_1_LM_diff_leq_10.list --output_dir=data/train/LM_diff_10/ --label_dir=data/train/wenet_1k_untar_pred_5_27_LM_diff_leq_10/labels/
# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/data_wenet_1k_split_100/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet1k_good_split_60/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet_4khr_split_60/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet1k_redo_split_60/


def get_args():
    parser = argparse.ArgumentParser(description='sum up prediction wer')
    parser.add_argument('--job_nums',type=int, default=8, help='number of total split dir')
    parser.add_argument('--data_list_dir',required=True, help='the path to the data_list dir eg data/train/wenet1k_good_split_60/')
    parser.add_argument('--label', type=bool, default= False, help = 'if ture, label file will also be considered.')
    parser.add_argument('--hypothesis', type=str, required=True, help='the hypothesis path.  eg. /hypothesis_0.txt ')
    parser.add_argument('--wav_dir', type=str, required=True, help='the wav dir path.  eg. data/train/wenet_1k_untar/ ')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_list_dir = args.data_list_dir
    num_lists = args.job_nums
    hypo = args.hypothesis
    # wav_dir is the directory where your pair of ID.scp (the audio file ) and ID.txt (the optional label file ) file stored. We assumed that you have
    # generated this dir in data processing steps.
    wav_dir = args.wav_dir

    print("data_list_path is", data_list_dir)
    print("num_lists is", num_lists)
    print("hypo is", hypo)
    print("wav_dir is", wav_dir)

    i = num_lists
    c = 0

    hypo_path = data_list_dir + "data_sublist" + str(i) + hypo
    output_wav = data_list_dir + "data_sublist" + str(i) + "/wav.scp"
    output_label = data_list_dir + "data_sublist" + str(i) + "/label.txt"
    # bad lines are just for debugging
    output_bad_lines = data_list_dir + "data_sublist" + str(i) + "/bad_line.txt"

    with open(hypo_path, 'r', encoding="utf-8") as reader:
        hypo_lines = reader.readlines()

    wavs = []
    labels = []
    bad_files = []
    for x in hypo_lines:
        c += 1
        file_id = x.split()[0]

        label_path = wav_dir + file_id + ".txt"
        wav_path = wav_dir + file_id + ".wav\n"
        wav_line = file_id + " " + wav_path
        wavs.append(wav_line)

        try:
            with open(label_path, 'r', encoding="utf-8") as reader1:
                label_line = reader1.readline()
        except OSError as e:
            bad_files.append(label_path)

        label_line = file_id + " " + label_line + "\n"
        labels.append(label_line)

    with open(output_wav, 'w', encoding="utf-8") as writer2:
        for wav in wavs:
            writer2.write(wav)
    with open(output_label, 'w', encoding="utf-8") as writer3:
        for label in labels:
            writer3.write(label)
    with open(output_bad_lines, 'w', encoding="utf-8") as writer4:
        for line in bad_files:
            writer4.write(line)

if __name__ == '__main__':
    main()