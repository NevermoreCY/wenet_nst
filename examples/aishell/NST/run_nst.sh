#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an augmented version of aishell-1 "run.sh" to make the code compatible with noisy student training

. ./path.sh || exit 1;
# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="0"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=1 # start from 0 if you need to start from data preparation
stop_stage=8

# here are extra parameters used in NST
cer_out_dir=""
dir=""
pseudo_data_list=""
supervised_data_list=""
gcmvn=""
checkpoint=
data_list=""
hypo_name=""
out_data_list=""
#parameters with default values:
label=0
average_num=30
nj=16
num_split=1
cer_hypo_threshold=10
speak_rate_threshold=0
label_file="label.txt"
utter_time_file="utter_time.json"
enable_nst=1
job_num=0
dir_split="wenet_split_60_test/"
hypo_name="hypothesis_nst${job_num}.txt"
wav_dir="data/train/wenet_1k_untar/"
tar_dir="data/train/wenet_1khr_tar/"
untar_dir="data/train/wenet_1khr_untar/"
cer_hypo_dir="wenet_cer_hypo"
cer_label_dir="wenet_cer_label"

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.

num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_ranHk=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0


dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard
num_utts_per_shard=1000

train_set=train

train_config=conf/train_conformer_nst.yaml

cmvn=true
#dir=exp/conformer_wenet1k_nst5_LM_diff_leq_10
#checkpoint=exp/conformer_wenet1k_nst0_sr3_v2/avg_30.pt

# use average_checkpoint will get better result

average_checkpoint=true
target_pt=80
decode_checkpoint=$dir/$target_pt.pt

# here we only use attention_rescoring for NST
decode_modes="attention_rescoring"
#decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"


. tools/parse_options.sh || exit 1;

# print the settings
echo "setting for this run:"
echo "dir is ${dir}"
echo "pseudo data list is ${pseudo_data_list}"
echo "supervised data list is ${supervised_data_list}"
echo "job_num is ${job_num}"
echo "cer_out_dir is  ${cer_out_dir}"
echo "average_num is ${average_num}"
echo "checkpoint is ${checkpoint} "
echo "enable_nst is ${enable_nst} "

# Data preparation
# we assumed that you have finished the data pre-process steps from -1 to 3 in aishell1/s0/run.sh .
# You can modify the "--supervised_data_list" to match your supervised data list.
# Here i used wenetspeech as the unsupervised data, you can run the data pre-process steps from -1 to 3 in
# wenetspeech/s0/run.sh ; you can modify "--pseudo_data_list" to match your unsupervised data list.
# In guideline, We extracted 1khr data from WenetSpeech and data should be prepared and stored in the following format:
# data.list files contains paths for all the extracted wenetspeech data and AISHELL-1 data.
# For unsupervised data, all the audio data (id.wav) and labels (id.txt which is optional) should be stored in wav_dir.
# A Json file containing audio lengths named as "utter_time.json" if you want to apply the speaking rate filter.
# we include a tiny example data under local/example to make it clearer for reproduction.
# you can follow this format to generate your own dataset as well.




# stage 1 is for training

# when training the initial fully supervised teacher, you need to set "--enable_nst 0".
# Example usage for initial teacher training :
# bash run_nst.sh  --stage 1 --stop-stage 1 --dir exp/conformer_nst_0
# --supervised_data_list data_aishell.list --enable_nst 0

# when training with pseudo-label in NST iterations, you need to set "--enable_nst 1".
# Example usage for NST iterations :
# bash run_nst.sh  --stage 1 --stop-stage 1 --dir exp/conformer_nst_1
# --supervised_data_list data_aishell.list --pseudo_data_list wenet_1khr_nst0.list --enable_nst 1

# "--dir" is the directory that stores the training parameters.
# "--supervised_data_list" is the data.list file for your supervised data.
# "--pseudo_data_list" is the data.list file for your pseudo-label.
# "--enable_nst" set to 0 will only use supervised data, if set to 1, pseudo-label will also be used in training.
# The ratio between supervised data and pseudo-label can be set in config file.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "********stage 1 start time : $now ********"
  mkdir -p $dir
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  rm $dir/ddp_init
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"

  # the global_cmvn file need to be calculated by combining both supervised/unsupervised datasets,
  # and it should be positioned at data/${train_set}/global_cmvn .
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir/global_cmvn
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  echo "checkpoint is "  ${checkpoint}
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    echo "gpu number  $i "
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.

    rank=`expr $node_rank \* $num_gpus + $i`

    # "--train_data_supervised" is the name of datalist for supervised data, which refers to aishell-1 in the example.
    # "--train_data_unsupervised" is the name of datalist for unsupervised data, which refers to wenetSpeech here.
    # both list should be stored under the train dir.
    # For supervised training, one could either set "--enable_nst" to 0 ,
    # or set the config pseudo-ratio to 0 so that none of the pseudo data is used.
    # For NST training, keep "--enable_nst" = 1,

    python wenet/bin/train_nst.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data_supervised data/$train_set/$supervised_data_list \
      --train_data_unsupervised data/$train_set/$pseudo_data_list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      --enable_nst $enable_nst \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi

# In stage 2, we get the averaged final checkpoint and calculate the test and dev accuracy.
# please make sure your test and valid data.list are in the proper location.

# here is an example usage:
# run_nst.sh  --stage 2 --stop-stage 2 --dir exp/conformer_nst_0 --supervised_data_list data_aishell.list --enable_nst 0

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # stage 5 we test with aishell dataset,
  echo "******** stage 2 start time : $now ********"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi

  # export model
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  # test_wer
  for mode in ${decode_modes}; do
  {
    #test_dir=$dir/test_${mode}_${target_pt}pt  # for target pt
    test_dir=$dir/test_${mode}${average_num}pt   # for average pt
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "before compute-wer"
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text > $test_dir/wer
  } &
  done

#   dev_wer
  for mode in ${decode_modes}; do
  {
    #test_dir=$dir/test_${mode}_${target_pt}pt  # for target pt
    dev_dir=$dir/dev_${mode}${average_num}pt   # for average pt
    mkdir -p $dev_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/dev/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $dev_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "before compute-wer"
    python tools/compute-wer.py --char=1 --v=1 \
      data/dev/text $dev_dir/text > $dev_dir/wer
  } &
  done
  wait
fi



# In stage 3, we Split the unsupervised data list to N parts, we will do parallel computing on those parts. Note we only
# need to perform stage 3 once in the intial teacher training, all the NST iterations afterwards will ignore stage 3.

# Example usage:
# bash run_nst.sh  --stage 3 --stop-stage 3 --num_split 1 --data_list wenet_1khr.list  --dir_split wenet_split_60

# "--num_split" is the number of sub-data we will split. The default split number is 1 which means you don't split data.
# In our paper's experiment, We used num_split = 60 which saved us lots of inference time & data shards generation time.
# "--data_list" is the data list file for unsupervised data.
# "--dir_split" is the directory for storing split unsupervised data.

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${enable_nst} -eq 0 ]; then
  echo "********stage 3 start time : $now ********"
  python split_data_list.py \
    --job_nums $num_split \
    --data_list_path data/train/$data_list \
    --output_dir data/train/$dir_split

fi


# stage 4 will perform inference without language model on the given sublist(job num).

# Example usage:
# bash run_nst.sh --stage 4 --stop-stage 4 --job_num 0 --dir_split wenet_split_60/
# --hypo_name hypothesis_nst0.txt --dir exp/conformer_nst_0

# "--job_num" specifies which sub-data split in stage 3 will be used. "job_num" must be less than "num_split".
# "--hypo_name" is the path for output hypothesis under the split directory.
# For each gpu/cpu, you can run with different job_num to perform data-wise parallel computing.

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "********stage 4 start time : $now ********"
  # we assume you have run stage 2 so that avg_${average_num}.pt exists
  decode_checkpoint=$dir/avg_${average_num}.pt
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  mode="attention_rescoring"
  gpu_id=0
  echo "job number  ${job_num} "
  echo "data_list dir is  ${dir_split}"
  echo "hypo name is " $hypo_name
  echo "dir is ${dir}"

  python wenet/bin/recognize.py --gpu $gpu_id \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/train/${dir_split}data_sublist${job_num}/data_list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file data/train/${dir_split}data_sublist${job_num}/${hypo_name} \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "end time : $now"

fi


# Stage 5 will generate wav.scp file and label.txt file(optional) for each sublist we generated in step 3. Note we only
# need to perform stage 5 once in the initial teacher training, all the NST iterations afterwards will ignore stage 5.

# Example usage:
# bash run_nst.sh --stage 5 --stop-stage 5 --job_num 0 --dir_split wenet_split_60_test/
# --hypo_name hypothesis_0.txt --label 1 --wav_dir data/train/wenet_1k_untar/

# "--label " sets to 1 means you have labels for unsupervised data, other wise set it to 0.
# "--wav_dir" is the directory that stores raw audio file(id.wav) and possible labels (id.txt).
# For each gpu or cpu, you can run with different job_num to perform data-wise parallel computing.

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ ${enable_nst} -eq 0 ]; then
  echo "********stage 5 start time : $now ********"
  python get_wav_labels.py \
    --dir_split data/train/${dir_split} \
    --hypo_name /$hypo_name \
    --wav_dir $wav_dir\
    --job_num $job_num \
    --label $label
fi

# stage 6 will calculate cer-hypo between hypothesis with and without language model.
# We assumed that you have trained a language model using the wenet aishell-1 pipline.
# (You should have data/lang/words.txt , data/lang/TLG.fst files ready.)

# Here is an exmaple usage:
# bash run_nst.sh --stage 6 --stop-stage 6 --job_num 0 --dir_split wenet_split_60_test/
# --cer_hypo_dir wenet1k_cer_hypo --hypo_name hypothesis_0.txt --dir exp/conformer_nst_0

# "--cer_hypo_dir" is the directory under "$dir/Hypo_LM_diff10/" that stores the cer_hypo for each sub-data.
# For each gpu, you can run with different job_num to perform data-wise parallel computing.

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "********stage 6 start time : $now ********"
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "dir_split is " data/train/${dir_split} "nj is" $nj "hypo_file is" $hypo_name "cer out is" $cer_hypo_dir "lm is 4gram"
  echo "dir is " $dir
  if [ ! -f data/train/${dir_split}data_sublist${job_num}/${hypo_name}  ]; then
  echo "text file does not exists"
  exit 1;
  fi

  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    data/train/${dir_split}data_sublist${job_num}/wav.scp data/train/${dir_split}data_sublist${job_num}/${hypo_name} $dir/final.zip \
    data/lang_test/words.txt $dir/Hypo_LM_diff10/${cer_hypo_dir}_${job_num}
  now=$(date +"%T")
  echo "end time : $now"
fi

# (optional, only run this stage if you have true label for unsupervised data.)
# stage 7 will calculate cer-label between true label and hypothesis with language model, it only runs when "--label"
# is set to 1. You can use the output cer to evaluate NST's performance.

# Here is an exmaple usage:
# bash run_nst.sh --stage 7 --stop-stage 7 --job_num 0 --data_list_dir wenet_split_60_test/
# --cer_label_dir wenet1k_cer_label --label_file label.txt --dir exp/conformer_nst_0 --label 1

# "--cer_label_dir" is the directory under "$dir/Hypo_LM_diff10/" that stores the cer_label for each sub-data.
# "--label_file" is the path for label.txt file under the split directory.

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && [ ${label} -eq 1 ]; then
  echo "********stage 7 start time : $now ********"
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "dir_split is " data/train/${dir_split} "nj is" $nj "label_file is" $label_file "cer out is" $cer_label_dir "lm is 4gram"
  echo "dir is " $dir
  echo "label_file " data/train/${dir_split}data_sublist${job_num}/${label_file}
  if [ ! -f data/train/${dir_split}data_sublist${job_num}/${label_file}  ]; then
  echo "text file does not exists"
  exit 1;
  fi

  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    data/train/${dir_split}data_sublist${job_num}/wav.scp data/train/${dir_split}data_sublist${job_num}/${label_file} $dir/final.zip \
    data/lang_test/words.txt $dir/Hypo_LM_diff10/${cer_label_dir}_${job_num}
  now=$(date +"%T")
  echo "end time : $now"
fi

# stage 8 will apply CER-hypo filter strategy to the generated pseudo-label, and it will generate new shards for the
# filter pseudo label and make it available for next NST iteration.

# Here is an exmaple usage:
# python generate_filtered_pseudo_label.py --dir_num=0 --cer_hypo_dir=wenet_supervised_cer_hypo
# --cer_hypo_threshold=10 --speak_rate_threshold=0 --utter_time_file=utter_time.json
# --dir=exp/conformer_test_fully_supervised --untar_dir=data/train/wenet_step8/
# --tar_dir=data/train/wenet1k_tar_step8/ --wav_dir=data/train/wenet_1k_untar/
# --out_data_list data/train/wenet_1khr.list

# "--cer_hypo_threshold" is the threshold for cer-hypo filter, default is set to 10 which means data with cer-hypo > 10
# will be pruned.
# "--speak_rate_threshold" is the threshold for speaking rate filter,
# data with speaking rate < threshold will be pruned.
# "--utter_time_file" is the json file that contains audio length information, the format can be found in local/example
# "--untar_dir" is the directory that stores the raw pseudo-label.
# "--tar_dir" is the directory that stores the shards of pseudo-label, this will be used for training in next iteration.
# "--out_data_list" is the data.list file contains the path to the shards in $tar_dir. This data.list file is the final
# output of current NST iteration and it will become the psuedo-data list in stage 1 for next iteration.

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "********stage 8 start time : $now ********"
  python local/generate_filtered_pseudo_label.py  \
    --cer_hypo_dir $cer_hypo_dir \
    --untar_dir $untar_dir \
    --wav_dir $wav_dir \
    --dir_num $job_num \
    --cer_hypo_threshold $cer_hypo_threshold\
    --speak_rate_threshold $speak_rate_threshold \
    --dir $dir \
    --tar_dir $tar_dir \
    --utter_time_file $utter_time_file

  python local/generate_data_list.py  \
    --tar_dir $tar_dir \
    --out_data_list $out_data_list

fi



