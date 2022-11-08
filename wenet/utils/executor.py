# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
import random
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import time

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader_aishell, data_loader_wenetspeech, device, writer,
              args, scaler, dataset_num, fuse_batch):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0


        if fuse_batch:
            with model_context():
                # --------------------------modification start ------------------------------------------
                print(time.time(), "before loop")
                iter_wenet = iter(data_loader_wenetspeech)

                for batch_idx, batch in enumerate(data_loader_aishell):
                    print(time.time(),"batch id is ", batch_idx)
                    # get data from aishell
                    key, feats, target, feats_lengths, target_lengths = batch
                    #log_str = "Time after feature v-stack " + str(time.time())
                    #print("dataset_num" , dataset_num, dataset_num == 2 )
                    #logging.info("wtfffffffffffffffff1111" , dataset_num == 2)
                    if dataset_num == 2:
                        #logging.info("wtfffffffffffffffff2222")
                        # get data from wenetspeech
                        key2,feats2,target2,feats_lengths2,target_lengths2 = next(iter_wenet)
                        #print("time after loading", time.time())
                        #x1 = feats_lengths.max()
                        #print("testing print, # of item in each batch", len(batch), len(next_wenet_batch))
                        #print("aishell batch:/n")
                        #print("ai key is" , key)
                        #print("we key is" , key2)
                        # key_c = key + key2
                        #print("key_c is", key_c)

                        #print("ai feats is", feats.shape, "we feats is", feats2.shape)
                        b_a,r_a,f_a = feats.shape
                        b_w,r_w,f_w = feats2.shape

                        #print( "Time before padding" ,time.time())
                        if r_a > r_w:
                            diff = r_a - r_w
                        #   print(diff)
                            feats2 = F.pad(input = feats2, pad = (0,0,0,diff), mode = 'constant', value = 0)
                        else:
                            diff = r_w - r_a
                            feats = F.pad(input = feats, pad = (0,0,0,diff), mode = 'constant', value = 0)
                        #print("time after padding", time.time())

                        #print("Time after padding", time.time())
                        #print("ai feats is", feats.shape, "we feats is", feats2.shape)

                        #print("Time before feature v-stack ", time.time())

                        feats_c = torch.cat([feats,feats2],dim=0)


                        # log_str = "Time after feature v-stack "+ str(time.time())
                        # logging.debug(log_str)

                        #print("feats_c is", feats_c.shape)
                        #new_data = F.pad(input = feats, pad = (0,0,0,2), mode = 'constant', value = 0)
                        #print("after padding", new_data.shape)
                        #b_fc,r_fc,f_fc = feats_c.shape
                        #for b in feats_c:
                         #   for x in range(r_fc):
                         #       print(x, b[x].sum())

                      #  print("we feat is ", feats2.shape, feats2)
                        #print("target is", target.shape, target)
                        #print("we target is ", target2.shape, target2)
                        t1a,t2a = target.shape
                        t1w,t2w = target2.shape
                        if t2a > t2w:
                            diff = t2a -t2w
                            target2 = F.pad(input = target2, pad = (0,diff), mode = 'constant', value = -1)
                        else:
                            diff = t2w -t2a
                            target = F.pad(input = target, pad = (0,diff), mode = 'constant', value = -1)

                        #print("Time before length stack ", time.time())
                        target_c = torch.vstack((target,target2))

                        #print("target_c os", target_c.shape, target_c)
                        #print("feats_lengths is", feats_lengths)
                        #print("we feats len is",feats_lengths2)
                        feats_length_c = torch.hstack((feats_lengths, feats_lengths2))
                        #print("feats length c", feats_length_c.shape, feats_length_c)
                        #print("target_lengths is", target_lengths)
                        #print("target_lengths 2 is", target_lengths2)
                        target_length_c = torch.hstack((target_lengths, target_lengths2))
                        #print("Time after length stack ", time.time())
                        #print("target length c", target_length_c.shape, target_length_c)
                        #print("wenetspeech batch",next_wenet_batch)
                        print("Time after feature v-stack " , time.time())

                        #logging.info("wtfffffffffffffffff3333")
                        #logging.info(log_str2)

                        #--------------------------modification end------------------------------------------
                        feats = feats_c
                        target = target_c
                        feats_lengths = feats_length_c
                        target_lengths = target_length_c

                    feats = feats.to(device)
                    target = target.to(device)
                    feats_lengths = feats_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                    #print("Time after to device ", time.time())
                    num_utts = target_lengths.size(0)
                    if num_utts == 0:
                        continue
                    context = None
                    # Disable gradient synchronizations across DDP processes.
                    # Within this context, gradients will be accumulated on module
                    # variables, which will later be synchronized.
                    if is_distributed and batch_idx % accum_grad != 0:
                        context = model.no_sync
                    # Used for single gpu training and DDP gradient synchronization
                    # processes.
                    else:
                        context = nullcontext
                    with context():
                        # autocast context
                        # The more details about amp can be found in
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with torch.cuda.amp.autocast(scaler is not None):
                            loss, loss_att, loss_ctc = model(
                                feats, feats_lengths, target, target_lengths)
                            loss = loss / accum_grad
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    num_seen_utts += num_utts
                    if batch_idx % accum_grad == 0:
                        if rank == 0 and writer is not None:
                            writer.add_scalar('train_loss', loss, self.step)
                        # Use mixed precision training
                        if use_amp:
                            scaler.unscale_(optimizer)
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            # Must invoke scaler.update() if unscale_() is used in
                            # the iteration to avoid the following error:
                            #   RuntimeError: unscale_() has already been called
                            #   on this optimizer since the last update().
                            # We don't check grad here since that if the gradient
                            # has inf/nan values, scaler.step will skip
                            # optimizer.step().
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            if torch.isfinite(grad_norm):
                                optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        self.step += 1
                    if batch_idx % log_interval == 0:
                        lr = optimizer.param_groups[0]['lr']
                        log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                            epoch, batch_idx,
                            loss.item() * accum_grad)
                        if loss_att is not None:
                            log_str += 'loss_att {:.6f} '.format(loss_att.item())
                        if loss_ctc is not None:
                            log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                        log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                        logging.debug(log_str)
        else:
            # case we are not fusing batch
            pseudo_ratio = float(args["pseudo_ratio"])
            print("************** not using batch fusion, pseudo_ratio is" , pseudo_ratio, "********************")
            with model_context():
                # --------------------------modification start ------------------------------------------
                print(time.time(), "before loop")
                iter_wenet = iter(data_loader_wenetspeech)
                iter_aishell = iter(data_loader_aishell)
                batch_idx = 0
                dl_weight = [pseudo_ratio, 1-pseudo_ratio]
                #rng = random.Random
                while True:

                    #print("batch id is ",batch_idx , "Time is ", time.time())
                    r = random.choices(population=[0,1], weights=dl_weight, k=1)[0]
                    # print(now is random)
                    dl = iter_wenet if r == 0 else iter_aishell
                    try:
                        key, feats, target, feats_lengths, target_lengths = next(dl)
                    except StopIteration:
                        name = "wenet_speech" if r == 0 else "aishell"
                        logging.info(f"{name} reaches end of dataloader for rank {rank}")
                        break
                    #name = "wenet" if r <= pseudo_ratio else "aishell"
                    #print("current dataset : " , name )

                    #print("finish random time ", time.time())
                    batch_idx += 1



                    feats = feats.to(device)
                    target = target.to(device)
                    feats_lengths = feats_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                    # print("Time after to device ", time.time())
                    num_utts = target_lengths.size(0)
                    if num_utts == 0:
                        continue
                    context = None
                    # Disable gradient synchronizations across DDP processes.
                    # Within this context, gradients will be accumulated on module
                    # variables, which will later be synchronized.
                    if is_distributed and batch_idx % accum_grad != 0:
                        context = model.no_sync
                    # Used for single gpu training and DDP gradient synchronization
                    # processes.
                    else:
                        context = nullcontext
                    with context():
                        # autocast context
                        # The more details about amp can be found in
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with torch.cuda.amp.autocast(scaler is not None):
                            loss, loss_att, loss_ctc = model(
                                feats, feats_lengths, target, target_lengths)
                            loss = loss / accum_grad
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    num_seen_utts += num_utts
                    if batch_idx % accum_grad == 0:
                        if rank == 0 and writer is not None:
                            writer.add_scalar('train_loss', loss, self.step)
                        # Use mixed precision training
                        if use_amp:
                            scaler.unscale_(optimizer)
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            # Must invoke scaler.update() if unscale_() is used in
                            # the iteration to avoid the following error:
                            #   RuntimeError: unscale_() has already been called
                            #   on this optimizer since the last update().
                            # We don't check grad here since that if the gradient
                            # has inf/nan values, scaler.step will skip
                            # optimizer.step().
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = clip_grad_norm_(model.parameters(), clip)
                            if torch.isfinite(grad_norm):
                                optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        self.step += 1
                    if batch_idx % log_interval == 0:
                        lr = optimizer.param_groups[0]['lr']
                        log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                            epoch, batch_idx,
                            loss.item() * accum_grad)
                        if loss_att is not None:
                            log_str += 'loss_att {:.6f} '.format(loss_att.item())
                        if loss_ctc is not None:
                            log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                        log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                        logging.debug(log_str)



    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                #print("cv debug , batch_idx is, " , batch_idx, "rank is", rank, "epoch is", epoch )
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts














