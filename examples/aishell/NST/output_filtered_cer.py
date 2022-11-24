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
    parser = argparse.ArgumentParser(description='out_put predictions')
    parser.add_argument('--dir_num',required=True,help='directory number')

    parser.add_argument('--utter_time_file', required=True, help='the cer threshold we use to filter')
    parser.add_argument('--cer_pred_threshold', required=True,type=float, help='the cer threshold we use to filter')
    parser.add_argument('--speak_rate_threshold', required=True,type=float, help='the cer threshold we use to filter')
    # path for cer
    parser.add_argument('--cer_label_dir',required=True, help='number of utterence per shard')
    parser.add_argument('--cer_hypo_dir', required=True,   help='the cer threshold we use to filter')
    # output untar and tar
    parser.add_argument('--output_dir',required=True, help='the output path, eg: data/train/wenet_untar_6_5_cer_hypo_leq_10_nst1/')
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
    cer_label_dir = args.cer_label_dir
    cer_hypo_dir = args.cer_hypo_dir
    output_dir = args.output_dir
    cer_pred_threshold = args.cer_pred_threshold
    speak_rate_threshold = args.speak_rate_threshold
    utter_time_file = args.utter_time_file  # utter_time_file = 'utter_time.json'
    tar_dir = args.tar_dir
    wav_dir = args.wav_dir
    start_tar_id = args.start_tar_id
    # wav_dir = "data/train/wenet_1k_untar/"
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print("start tar id is", start_tar_id)
    print("make dirs")

    # prediction_lm_dir = "test_data/wenet_1k_cer_hypo_nst3/wer"
    # label_lm_dir = "test_data/wenet_1k_cer_label_nst3/wer"
    utter_time_enable = True
    dataset = "wenet"
    # wav_dir = "data/train/wenet_1k_untar/"

    utter_time = {}
    if utter_time_enable:

        if dataset == "wenet":
            print("wenet")
            # utter_time_file = 'utter_time.json'

            with open(utter_time_file, encoding='utf-8') as fh:
                utter_time = json.load(fh)

        if dataset == "aishell2":
            # aishell2_jason = "test_data/train_manifest.json"
            aishell2_jason = utter_time_file
            c = 0
            print("aishell2")

            with open(aishell2_jason, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    data_text = data["text"]
                    data_audio = data["audio_filepath"]

                    t_id = data_audio.split("/")[-1].split(".")[0]
                    data_duration = data["duration"]
                    # data_audio = "/" + data_audio[16:]
                    # data_list.append([data_text,data_audio,t_id, data_duration])
                    utter_time[t_id] = data_duration
                    # data_list.append([t_id, data_duration])

    print(time.time(), "start time ")


    cer_dict = {}

    print("dir_num = ", dir_num)
    cer_hypo_path = cer_hypo_dir + str(dir_num)
    cer_label_path = cer_label_dir + str(dir_num)
    with open(cer_hypo_path, 'r', encoding="utf-8") as reader:
        data = reader.readlines()

    L = len(data)
    for i in range(L):

        line = data[i]
        # print(i)
        # print(line)
        if line[:3] == 'utt':
            wer_list = data[i + 1].split()
            wer_pred_lm = float(wer_list[1])
            N_hypo = int(wer_list[3].split("=")[1])
            # print(wer)

            utt_list = line.split()
            lab_list = data[i + 2].split()
            rec_list = data[i + 3].split()
            # print(utt_list )
            # print(wer_list)
            # print(lab_list)
            # print(rec_list)
            utt_id = utt_list[1]
            pred_no_lm = "".join(lab_list[1:])
            pred_lm = "".join(rec_list[1:])

            prediction = "".join(lab_list[1:])
            # print(utt_id)
            # print(label)
            # print(pred)
            if utter_time_enable:

                utt_time = utter_time[utt_id]
                utt_rate = N_hypo / utt_time

                cer_dict[utt_id] = [pred_no_lm, pred_lm, wer_pred_lm, utt_time, N_hypo, prediction]
            else:
                cer_dict[utt_id] = [pred_no_lm, pred_lm, wer_pred_lm, -1, -1, prediction]
                #
                # filtered_line = [utt_id,label,pred]
                # data_filtered.append(filtered_line)

        # read data with true label
    with open(cer_label_path, 'r', encoding="utf-8") as reader:
        data = reader.readlines()
    L = len(data)
    for i in range(L):
        line = data[i]
        if line[:3] == 'utt':
            wer_list = data[i + 1].split()
            wer_label_lm = float(wer_list[1])
            N = int(wer_list[3].split("=")[1])
            C = int(wer_list[4].split("=")[1])
            S = int(wer_list[5].split("=")[1])
            D = int(wer_list[6].split("=")[1])
            I = int(wer_list[7].split("=")[1])
            # print(wer_list)
            # print(wer)

            utt_list = line.split()

            lab_list = data[i + 2].split()
            rec_list = data[i + 3].split()
            # print(utt_list )
            # print(wer_list)
            # print(lab_list)
            # print(rec_list)
            utt_id = utt_list[1]
            label_len = len(lab_list[1:])
            label = "".join(lab_list[1:])

            pred_lm = "".join(rec_list[1:])

            # print(utt_id)
            # print(label)
            # print(pred)
            if utt_id not in cer_dict:
                continue
            cer_dict[utt_id].append(label)
            cer_dict[utt_id].append(wer_label_lm)
            # print([N,C,S,D,I])
            cer_dict[utt_id].extend([N, C, S, D, I])
    c = 0
    cer_label_total = 0
    cer_labels = []
    cer_preds = []
    uttr_len = []
    speak_rates = []
    total_N = 0
    total_C = 0
    total_S = 0
    total_D = 0
    total_I = 0
    num_lines = 0
    high_cer_label = []
    high_cer_label_speak_rate = []
    high_cer_label_N = []

    data_filtered = []

    for key, item in cer_dict.items():
        if len(item) < 7:
            continue
        cer_pred = item[2]
        cer_label = item[7]
        label_len = item[8]
        hypo_len = item[3]
        speak_rate = item[4] / item[3]  # char per second

        # cer_label_total += cer_label
        # c += 1
        # N = item[5]
        if cer_pred <= cer_pred_threshold and speak_rate > speak_rate_threshold:
            num_lines += 1
            cer_label_total += cer_label
            c += 1
            cer_labels.append(cer_label)
            cer_preds.append(cer_pred)
            uttr_len.append(item[4])
            speak_rates.append(speak_rate)
            N, C, S, D, I = item[8:13]
            total_N += N
            total_C += C
            total_S += S
            total_D += D
            total_I += I
            label = item[6]
            pred = item[1]
            utt_id = key
            filtered_line = [utt_id, label, pred]
            data_filtered.append(filtered_line)


    num_uttr = 1000
    L = len(data_filtered)
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
        out_put_label_dir = output_dir + "labels/"
        os.makedirs(out_put_label_dir, exist_ok=True)
        output_labels_path = out_put_label_dir+ "dir" + str(dir_num) + "_" + "tar" + tar_s + "_label.txt"
        with open(output_labels_path, "w", encoding="utf-8") as writer1:
            for i in range(cur_id,end_id):
                print("dir:",dir_num,", " "tar: ", tar_id , ", ", "progress:", i/L)

                t_id,label,utter = data_filtered[i]
                writer1.write(t_id + " " + label + '\n')

                output_path = out_put_dir + t_id + ".txt"
                wav_path = wav_dir + t_id + ".wav"
                print(wav_path)
                wav_exist = os.path.exists(wav_path)
                if wav_exist:
                    #print(t_id, utter, output_path)
                    # update .txt
                    with open(output_path,"w", encoding= "utf-8") as writer:
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





