import os
import random
import tarfile
import argparse


#merged_step 1:

#bash run_nst.sh  --stage 5 --stop-stage 5 --dir exp/conformer_test_fully_supervised --supervised_data_list data_aishell.list --enable_nst 0 --num_split 1 --data_list wenet_1khr.list --dir_split wenet_split_60_test/ --job_num 0 --hypo_name hypothesis_nst0.txt --label 1 --wav_dir data/train/wenet_1k_untar/ --cer_hypo_dir wenet_cer_hypo --cer_label_dir wenet_cer_label --label_file label.txt --cer_hypo_threshold 10 --speak_rate_threshold 0 --utter_time_file utter_time.json --untar_dir data/train/wenet_1khr_untar/ --tar_dir data/train/wenet_1khr_tar/ --out_data_list data/train/wenet_1khr.list 2>&1 | tee logs/log_12_4_step5


#bash run_nst.sh  --stage 1 --stop-stage 1 --dir exp/conformer_test_fully_supervised
# --supervised_data_list data_aishell.list --enable_nst 0 --num_split 1 --data_list wenet_1khr.list
# --dir_split wenet_split_60_test/ --job_num 0 --hypo_name hypothesis_nst0.txt --label 1 --wav_dir data/train/wenet_1k_untar/
# --cer_hypo_dir wenet_cer_hypo --cer_label_dir wenet_cer_label --label_file label.txt --cer_hypo_threshold 10 --speak_rate_threshold 0
# --utter_time_file utter_time.json --untar_dir data/train/wenet_1khr_untar/ --tar_dir data/train/wenet_1khr_tar/ --out_data_list data/train/wenet_1khr.list


# python split_data_list_nst.py --job_nums=50 --data_list_path=data/train/pseudo_data_iter_1_LM_diff_leq_10.list --output_dir=data/train/LM_diff_10/ --label_dir=data/train/wenet_1k_untar_pred_5_27_LM_diff_leq_10/labels/
# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/data_wenet_1k_split_100/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet1k_good_split_60/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet_4khr_split_60/

# python generate_wav_labels_without_label_dir.py --job_nums=0 --data_list_dir=data/train/wenet1k_redo_split_60/
#bash run_nst.sh --stage 5 --stop-stage 5 --job_num 0 --data_list_dir data/train/wenet_split_60_test/ --hypo_name hypothesis_0.txt --label false --wav_dir data/train/wenet_1k_untar/

def get_args():
    parser = argparse.ArgumentParser(description='sum up prediction wer')
    parser.add_argument('--job_num',type=int, default=8, help='number of total split dir')
    parser.add_argument('--dir_split',required=True, help='the path to the data_list dir eg data/train/wenet1k_good_split_60/')
    parser.add_argument('--label', type=int, default= 0, help = 'if ture, label file will also be considered.')
    parser.add_argument('--hypo_name', type=str, required=True, help='the hypothesis path.  eg. /hypothesis_0.txt ')
    parser.add_argument('--wav_dir', type=str, required=True, help='the wav dir path.  eg. data/train/wenet_1k_untar/ ')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_list_dir = args.dir_split
    num_lists = args.job_num
    hypo = args.hypo_name
    # wav_dir is the directory where your pair of ID.scp (the audio file ) and ID.txt (the optional label file ) file stored. We assumed that you have
    # generated this dir in data processing steps.
    wav_dir = args.wav_dir
    label = args.label

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
        # print(c)
        c += 1
        file_id = x.split()[0]

        label_path = wav_dir + file_id + ".txt"
        wav_path = wav_dir + file_id + ".wav\n"
        wav_line = file_id + " " + wav_path
        wavs.append(wav_line)
        if label:
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
    with open(output_bad_lines, 'w', encoding="utf-8") as writer4:
        for line in bad_files:
            writer4.write(line)
    if label:
        with open(output_label, 'w', encoding="utf-8") as writer3:
            for label in labels:
                writer3.write(label)

if __name__ == '__main__':
    main()