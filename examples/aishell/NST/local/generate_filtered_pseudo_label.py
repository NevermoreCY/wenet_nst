import argparse
import os
import random
import tarfile
import time
import numpy as np
# import matplotlib.pyplot as plt
import json

# /yuch_ws/wenet/examples/aishell/s0/; python output_filtered_cer.py --dir_num=2 --cer_pred_threshold=10 --speak_rate_threshold=3 --utter_time_file=utter_time.json --cer_label_dir=test_data/wenet_1k_cer_label_nst0_redo/wer --cer_hypo_dir=test_data/wenet_1k_cer_hypo_nst0_redo/wer --output_dir=data/train/wenet1k_untar_redo_8_1_nst0/ --tar_dir=data/train/wenet1k_tar_redo_8_1_nst0/ --wav_dir=data/train/wenet_1k_untar/

def get_args():
    parser = argparse.ArgumentParser(description='generate filter pseudi label')
    parser.add_argument('--dir_num',required=True,help='split directory number')
    parser.add_argument('--cer_hypo_dir', required=True, help='prefix for cer_hypo_dir')
    parser.add_argument('--utter_time_file', required=True, help='the json file that contains audio time infos ')
    parser.add_argument('--cer_hypo_threshold', required=True,type=float, help='the cer-hypo threshold we use to filter')
    parser.add_argument('--speak_rate_threshold',type=float, help='the cer threshold we use to filter')
    parser.add_argument('--dir', required=True,   help='dir for the experiment ')
    # output untar and tar
    parser.add_argument('--untar_dir',required=True, help='the output path, eg: data/train/wenet_untar_6_5_cer_hypo_leq_10_nst1/')
    parser.add_argument('--tar_dir', required=True,
                        help='the tar file path, eg: data/train/wenet_tar_6_5_cer_hypo_leq_10_nst1/')
    parser.add_argument('--wav_dir', required=True, help='dir to store wav files, eg "data/train/wenet_1k_untar/"')
    parser.add_argument('--start_tar_id', default=0,type=int, help='the initial tar id (for debugging)')
    args = parser.parse_args()
    return args


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def main():
    args = get_args()

    dir_num = args.dir_num
    dir = args.dir
    output_dir = args.untar_dir
    cer_hypo_threshold = args.cer_hypo_threshold
    speak_rate_threshold = args.speak_rate_threshold
    utter_time_file = args.utter_time_file  # utter_time_file = 'utter_time.json'
    tar_dir = args.tar_dir
    wav_dir = args.wav_dir
    start_tar_id = args.start_tar_id
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    cer_hypo_name = args.cer_hypo_dir
    print("start tar id is", start_tar_id)
    print("make dirs")

    utter_time_enable = True
    dataset = "wenet"

    utter_time = {}
    if utter_time_enable:

        if dataset == "wenet":
            print("wenet")
            with open(utter_time_file, encoding='utf-8') as fh:
                utter_time = json.load(fh)

        if dataset == "aishell2":
            aishell2_jason = utter_time_file
            print("aishell2")
            with open(aishell2_jason, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    data_audio = data["audio_filepath"]
                    t_id = data_audio.split("/")[-1].split(".")[0]
                    data_duration = data["duration"]
                    utter_time[t_id] = data_duration


    print(time.time(), "start time ")


    cer_dict = {}

    print("dir_num = ", dir_num)
    cer_hypo_path = dir + "/Hypo_LM_diff10/" + cer_hypo_name+ "_" + dir_num + "/wer"
    with open(cer_hypo_path, 'r', encoding="utf-8") as reader:
        data = reader.readlines()

    L = len(data)
    for i in range(L):

        line = data[i]
        if line[:3] == 'utt':
            wer_list = data[i + 1].split()
            wer_pred_lm = float(wer_list[1])
            N_hypo = int(wer_list[3].split("=")[1])
            # print(wer)

            utt_list = line.split()
            lab_list = data[i + 2].split()
            rec_list = data[i + 3].split()

            utt_id = utt_list[1]
            pred_no_lm = "".join(lab_list[1:])
            pred_lm = "".join(rec_list[1:])
            prediction = "".join(lab_list[1:])

            if utter_time_enable:

                utt_time = utter_time[utt_id]

                cer_dict[utt_id] = [pred_no_lm, pred_lm, wer_pred_lm, utt_time, N_hypo, prediction]
            else:
                cer_dict[utt_id] = [pred_no_lm, pred_lm, wer_pred_lm, -1, -1, prediction]


    c = 0

    cer_preds = []
    uttr_len = []
    speak_rates = []
    num_lines = 0


    data_filtered = []

    for key, item in cer_dict.items():

        cer_pred = item[2]
        speak_rate = item[4] / item[3]  # char per second

        if cer_pred <= cer_hypo_threshold and speak_rate > speak_rate_threshold:

            num_lines += 1
            c += 1
            cer_preds.append(cer_pred)
            uttr_len.append(item[4])
            speak_rates.append(speak_rate)
            pred = item[1]
            utt_id = key
            filtered_line = [utt_id,  pred]
            data_filtered.append(filtered_line)


    num_uttr = 1000
    L = len(data_filtered)
    print("total sentences after filter ")
    cur_id = start_tar_id * 1000
    end_id = cur_id + num_uttr
    if cur_id < L and end_id > L:
        end_id = L
    tar_id = start_tar_id

    not_exist = []
    while end_id <= L:

        tar_s = str(tar_id)
        diff = 6 - len(tar_s)
        for _ in range(diff):
            tar_s = "0" + tar_s

        out_put_dir = output_dir + "dir" + str(dir_num) + "_" + "tar" + tar_s + "/"
        os.makedirs(out_put_dir, exist_ok=True)

        for i in range(cur_id, end_id):
            print("dir:", dir_num, ", " "tar: ", tar_id, ", ", "progress:", i / L)

            t_id, utter = data_filtered[i]

            output_path = out_put_dir + t_id + ".txt"
            wav_path = wav_dir + t_id + ".wav"
            print(wav_path)
            wav_exist = os.path.exists(wav_path)
            if wav_exist:
                # print(t_id, utter, output_path)
                # update .txt
                with open(output_path, "w", encoding="utf-8") as writer:
                    writer.write(utter)
                # update .wav
                os.system("cp" + " " + wav_path + " " + out_put_dir + t_id + ".wav")
            else:
                print(" wav does not exists ! ", wav_path)
                not_exist.append(wav_path)



        tar_file_name = tar_dir + "dir" + str(dir_num) + "_" + tar_s + ".tar"
        # tar the dir

        make_tarfile(tar_file_name, out_put_dir)
        # update index
        tar_id += 1
        cur_id += num_uttr
        end_id += num_uttr

        if cur_id < L and end_id > L:
            end_id = L

        print("end, now remove untar files")
        print("rm -rf" + " " + out_put_dir[:-1])
        os.system("rm -rf" + " " + out_put_dir[:-1])
        print("remove done")

    print("There are ", len(not_exist) , "wav files not exist")
    print(not_exist)





if __name__ == '__main__':
    main()





