# Introduction

# Guideline

## Data preparation:
You can run , we assumed that you follow the data pre-process steps from -1 to 3 in aishell1/s0/run.sh to prepare your dataset. 

For each train,valid and test set, a data.list file should be prepared, each line of the file is the path to the shards. 

For unsupervised data, all the audio datas (id.wav) and labels (id.txt which is optional) should be prepared and stored in data/train/wav_dir.

utter_time_file should 
## Initial supervised teacher:
``` sh
bash run_nst.sh --dir exp/conformer_test_fully_supervised --supervised_data_list data_aishell.list --data_list wenet_1khr.list --dir_split wenet_split_60_test/ --out_data_list data/train/wenet_1khr_nst0.list --enable_nst 0
```

The argument "dir" stores the training parameters, "supervised_data_list" contains paths for supervised data shards, "data_list" contains paths for unsupervised data shards which is used for inference. "dir_split" is the directory stores split unsupervised data for parallel computing. This guideline uses the default num_split equal to 1 while we strongly recommend use larger number to decrease the inference and shards generation time.  "out_data_list" is the pseudo label data list file path. "enable_nst" is whether we train with pseudo label, for initial teacher we set it to 0.

Full arguments are listed below, you can check the run_nst.sh code for more information about each stage and their arguments:

``` sh
bash run_nst.sh --stage 1 --stop-stage 8 --dir exp/conformer_test_fully_supervised --supervised_data_list data_aishell.list --enable_nst 0 --num_split 1 --data_list wenet_1khr.list --dir_split wenet_split_60_test/ --job_num 0 --hypo_name hypothesis_nst0.txt --label 1 --wav_dir data/train/wenet_1k_untar/ --cer_hypo_dir wenet_cer_hypo --cer_label_dir wenet_cer_label --label_file label.txt --cer_hypo_threshold 10 --speak_rate_threshold 0 --utter_time_file utter_time.json --untar_dir data/train/wenet_1khr_untar/ --tar_dir data/train/wenet_1khr_tar/ --out_data_list data/train/wenet_1khr.list 
```

## Noisy student interations:

After finishing the initial fully supervised baseline, we now have the pseudo-label data list which is "wenet_1khr_nst0.list" if you follow the guideline. We will use it as the pseudo_data in the training step and the pseudo-label for next NST iteration will be generated.

Here is an example code:

``` sh
bash run_nst.sh --dir exp/conformer_nst1 --supervised_data_list data_aishell.list --pseudo_data_list wenet_1khr_nst0.list  --enable_nst 1 --job_num 0 --hypo_name hypothesis_nst1.txt --untar_dir data/train/wenet_1khr_untar_nst1/ --tar_dir data/train/wenet_1khr_tar_nst1/ --out_data_list data/train/wenet_1khr_nst1.list 
```
Most of the arguments are same as the initial teacher training, here we add extra argument "pseudo_data_list" for path of pseudo data list. The enbale_nst must be set to 1 if you want to train with pseudo data. The index for hypo_name, tar_dir need to be changed if you don't want to overlap the previous generated data.
The output data list can be used as the input of pseudo-data list for next NST itearion.



Full arguments are listed below, you can check the run_nst.sh code for more information about each stage and their arguments:
``` sh
bash run_nst.sh --stage 1 --stop-stage 8 --dir exp/conformer_nst1 --supervised_data_list data_aishell.list --pseudo_data_list wenet_1khr_nst0  --enable_nst 1 --num_split 1 --dir_split wenet_split_60_test/ --job_num 0 --hypo_name hypothesis_nst1.txt --label 0 --wav_dir data/train/wenet_1k_untar/ --cer_hypo_dir wenet_cer_hypo --cer_label_dir wenet_cer_label --label_file label.txt --cer_hypo_threshold 10 --speak_rate_threshold 0 --utter_time_file utter_time.json --untar_dir data/train/wenet_1khr_untar_nst1/ --tar_dir data/train/wenet_1khr_tar_nst1/ --out_data_list data/train/wenet_1khr_nst1.list 
```
# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 8, 8 gpu, acc_grad 1, 100 epochs, dither 0.1
* Training weight info: transducer_weight 0.75, ctc_weight 0.1, attention_weight 0.15, average_num 10
* Predictor type: lstm

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.24  |

* after 165 epochs and avg 30

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.02  |
| ctc prefix beam search    | 5.17  |
| ctc prefix beam + rescore | 4.48  |

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 20, 8 gpu, acc_grad 1, 140 epochs, dither 0.1
* Training weight info: transducer_weight 0.4, ctc_weight 0.2, attention_weight 0.4, average_num 10
* Predictor type: lstm
* Model link: https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20220728_conformer_rnnt_exp.tar.gz

| decoding mode                         | CER   |
|---------------------------------------|-------|
| rnnt greedy search                    | 4.88  |
| rnnt beam search                      | 4.67  |
| ctc prefix beam search                | 5.02  |
| ctc prefix beam + rescore             | 4.51  |
| ctc prefix beam + rnnt&attn rescore   | 4.45  |
| rnnt prefix beam + rnnt&attn rescore  | 4.49  |


## U2++ Conformer Result

* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 4, 32 gpu, acc_grad 1, 360 epochs
* Training weight info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.15  average_num 30
* Predictor type: lstm

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| rnnt greedy search        | 5.68  | 6.26  |

## Pretrain
* Pretrain model: https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_exp.tar.gz
* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 8, 8 gpu, acc_grad 1, 140 epochs
* Training weight info: transducer_weight 0.4,  ctc_weight 0.2 , attention_weight 0.4, reverse_weight 0.3  average_num 30
* Predictor type: lstm

| decoding mode/chunk size    | full  | 16     |
|-----------------------------|-------|--------|
| rnnt greedy search          | 5.21  | 5.73   |
| rnnt prefix beam            | 5.14  | 5.63   |
| rnnt prefix beam + rescore  | 4.73  | 5.095  |


## Training loss ablation study

note:

- If rnnt is checked, greedy means rnnt  greedy search; so is beam

- if rnnt is checked, rescoring means rnnt beam & attention rescoring

- if only 'ctc & att' is checked, greedy means ctc gredy search; so is beam

- if only  'ctc & att' (AED)  is checked, rescoring means ctc beam & attention rescoring

- what if rnnt model do search of wenet's style, comming soon

| rnnt | ctc | att | greedy | beam | rescoring | fusion |
|------|-----|-----|--------|------|-----------|--------|
| ✔    | ✔   | ✔   |   4.88 | 4.67 |      4.45 |   4.49 |
| ✔    | ✔   |     |   5.56 | 5.46 |       /   |   5.40 |
| ✔    |     | ✔   |   5.03 | 4.94 |      4.87 |    /   |
| ✔    |     |     |   5.64 | 5.59 |       /   |    /   |
|      | ✔   | ✔   |   4.94 | 4.94 |      4.61 |    /   |
## Citations

``` bibtex

@article{chen2022NST,
  title={Improving Noisy Student Training on Non-target Domain Data for Automatic Speech Recognition},
  author={Chen, Yu and Wen, Ding and Lai, Junjie},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}