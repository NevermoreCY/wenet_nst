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
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# here are extra parameters used in NST
data_list_dir=""
job_num=-1
wav_label_dir=""
cer_out_dir=""
text_file=""
hypo_name=""
dir=""
pseudo_data_list=""
supervised_data_list=""
gcmvn=""
enable_nst=1
checkpoint=
average_num=30
nj=16
num_split=""
data_list=""
dir_split=""

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.

num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
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

#avg_30.pt

# use average_checkpoint will get better result
# ??
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
echo "data_list_dir is ${data_list_dir}"
echo "job_num is ${job_num}"
echo "wav_label_dir is ${wav_label_dir}"
echo "cer_out_dir is  ${cer_out_dir}"
echo "text_file is ${text_file}"
echo "average_num is ${average_num}"
echo "checkpoint is ${checkpoint} "
echo "enable_nst is ${enable_nst} "


# we assumed that you have finished the data pre-process steps from -1 to 3 in aishell1/s0/run.sh .
# You can modify the "--train_data_supervised" to match your supervised data list.
# Here i used wenetspeech as the unsupervised data, you can run the data pre-process steps from -1 to 3 in
# wenetspeech/s0/run.sh ; you can modify "--train_data_supervised" to match your unsupervised data list.
# you can follow this process to generate your own dataset.
# I have also included my code for extracting data in local/...

# stage 1 is for training
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
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

# In stage 2, we get the averaged final checkpoint and calculate the test and dev accuracy
# please make sure your test and valid data.list are in the proper location.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # stage 5 we test with aishell dataset,
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


# split the (unsupervised) datalist into N sublists, where N depends on the number of available cpu in your cluster.
# when making inference, we compute N sublist in parallel.
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  python split_data_list.py \
    --job_nums $num_split \
    --data_list_path data/train/$data_list \
    --output_dir data/train/$dir_split

fi


# stage 4 will perform inference without language model on the given sublist(job num)
# here is example usages:
# bash run_nst.sh --stage 4 --stop-stage 4 --job_num $i --data_list_dir data/train/wenet_4khr_split_60/
# --hypo_name hypothesis_nst4.txt --dir exp/conformer_aishell2_wenet4k_nst4
# You need to specify the "job_num" n (n <= N), "data_list_dir" which is the dir path for split data
# "hypo_name" is the path for output hypothesis and "dir" is the path where we train and store the model.
# For each gpu, you can run with different job_num to perform data-wise parallel computing.
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "start time : $now"
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
  echo "data_list dir is  ${data_list_dir}"
  echo "hypo name is " $hypo_name
  echo "dir is ${dir}"

  python wenet/bin/recognize.py --gpu $gpu_id \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data ${data_list_dir}data_sublist${job_num}/data_list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file ${data_list_dir}data_sublist${job_num}/${hypo_name} \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "end time : $now"

fi

# Calculate cer between hypothesis with and without language model
# Here is an exmaple usage:
# bash run_nst.sh --stage 5 --stop-stage 5 --job_num n --wav_label_dir data/train/wenet1k_redo_split_60/
# --cer_out_dir wenet1k_cer_hypo --text_file hypothesis_nst6.txt --dir exp/conformer_no_filter_redo_nst6
# You need to specify the "job_num" n (n <= N), "data_list_dir" which is the dir path for split data
# "hypo_name" is the path for output hypothesis and "dir" is the path where we train and store the model.
# For each gpu, you can run with different job_num to perform data-wise parallel computing.
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "wav_label_dir is " $wav_label_dir "nj is" $nj "text_file is" $text_file "cer out is" $cer_out_dir "lm is 4gram"
  echo "dir is " $dir
  if [ ! -f ${wav_label_dir}data_wenet${job_num}/${text_file}  ]; then
  echo "text file does not exists"
  exit 1;
  fi

  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    ${wav_label_dir}data_wenet${job_num}/wav.scp ${wav_label_dir}data_wenet${job_num}/${text_file} $dir/final.zip \
    data/lang_test/words.txt $dir/Hypo_LM_diff10/${cer_out_dir}_${job_num}
  now=$(date +"%T")
  echo "end time : $now"
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "wav_label_dir is " $wav_label_dir "nj is" 4 "text_file is" $text_file "cer out is" $cer_out_dir "lm is 4gram"
  echo "dir is " $dir

  ./tools/decode.sh --nj $nj \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0 --rescoring_weight 1 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test_4gram/TLG.fst \
    data/test/wav.scp data/test/text $dir/final.zip \
    data/lang_test_4gram/words.txt $dir/test_attention_rescoring_withLM_ctc0_rw1
  now=$(date +"%T")
  echo "end time : $now"
fi

