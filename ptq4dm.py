# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse


# quant
import os
from QDrop import quant

# from improved_diffusion import logger
# from improved_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
# )
from QDrop.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)
# from improved_diffusion.image_datasets import load_data
# from QDrop.data.imagenet import build_imagenet_data
import torch
import torch.nn as nn
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from torch.utils.data import DataLoader

import params

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'



from utils import plot_tensor, save_plot
from text.symbols import symbols


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale




def generate_t(args, t_mode, num_samples, num_timesteps, device):
    if t_mode == "1":
        t = torch.tensor([1] * num_samples, device=device)  # TODO timestep gen
    elif t_mode == "-1":
        # t = torch.tensor(
        #     [num_timesteps - 1] * num_samples, device=device
        # )  # TODO timestep gen
        t = torch.tensor([0.9] * num_samples, device=device)
    elif t_mode == "mean":
        t = torch.tensor(
            [num_timesteps // 2] * num_samples, device=device
        )/100  # TODO timestep gen
    elif t_mode == "manual":
        t = torch.tensor(
            [num_timesteps * 0.1] * num_samples, device=device
        )/100  # TODO timestep gen
    elif t_mode == "normal":
        shape = torch.Tensor(num_samples)
        normal_val = torch.nn.init.normal_(shape, mean=args.calib_t_mode_normal_mean, std=args.calib_t_mode_normal_std)*num_timesteps/100
        # t = normal_val.clone().type(torch.int).to(device=device)
        t = normal_val.clone().to(device=device)
        # print(t.shape)
        # print(t[0:30])
    elif t_mode == "random":
        # t = torch.randint(0, diffusion.num_timesteps, [num_samples], device=device)
        t = torch.randint(int(num_timesteps*0.5), int(num_timesteps), [num_samples], device=device)/100
        print(t.shape)
        print(t)
    elif t_mode == "uniform":
        t = torch.linspace(
            0, num_timesteps, num_samples, device=device
        ).round()/100
    else:
        raise NotImplementedError
    return t.clamp(0, num_timesteps - 1)


def quant_model(args, model, num_timesteps):
    # build quantization parameters
    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "scale_method": args.init_wmode,
        "symmetric": True,
    }
    aq_params = {
        "n_bits": args.n_bits_a,
        "channel_wise": False,
        "scale_method": args.init_amode,
        "leaf_param": True,
        "prob": args.prob,
        "symmetric": True,
    }

    qnn = QuantModel(
        model=model, weight_quant_params=wq_params, act_quant_params=aq_params
    )
    qnn.cuda()
    qnn.eval()
    print('qnn:',qnn)
    if not args.disable_8bit_head_stem:
        print("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()
    # # if args.mixup_quant_on_cosine:
    # print("Setting the cosine embedding layer to 32-bit")
    # qnn.set_cosine_embedding_layer_to_32bit()

    qnn.disable_network_output_quantization()
    # print("check the model!")
    # print(qnn)
    print("sampling calib data")
    if args.calib_im_mode == "random":
            cali_data = random_calib_data_generator(
                args,
                args.calib_num_samples,
                "cuda",
                args.calib_t_mode,
                num_timesteps,
            )
            
            
    elif args.calib_im_mode == "raw":
        cali_data = raw_calib_data_generator(
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            num_timesteps,
        )
    elif args.calib_im_mode == "raw_forward_t":
        cali_data = forward_t_calib_data_generator(
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            num_timesteps,
            model
        )
    elif args.calib_im_mode == "noise_backward_t":
        cali_data = backward_t_calib_data_generator(
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            num_timesteps,
            model,
        )
    else:
        raise NotImplementedError
    # print('the quantized model is below!')
    # Kwargs for weight rounding calibration
    assert args.wwq is True
    kwargs = dict(
        cali_data=cali_data,
        iters=args.iters_w,
        weight=args.weight,
        b_range=(args.b_start, args.b_end),
        warmup=args.warmup,
        opt_mode="mse",
        wwq=args.wwq,
        waq=args.waq,
        order=args.order,
        act_quant=args.act_quant,
        lr=args.lr,
        input_prob=args.input_prob,
        keep_gpu=not args.keep_cpu,
    )

    if args.act_quant and args.order == "before" and args.awq is False:
        """Case 2"""
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )

    """init weight quantizer"""
    set_weight_quantize_params(qnn)
    if not args.use_adaround:
        print('setting')
        # cali_data = cali_data.detach()
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )
        print('setting1111111')
        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn
    else:
        def set_weight_act_quantize_params(module):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, module, **kwargs)
            else:
                raise NotImplementedError

        def recon_model(model: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print("Reconstruction for layer {}".format(name))
                    set_weight_act_quantize_params(module)
                elif isinstance(module, BaseQuantBlock):
                    print("Reconstruction for block {}".format(name))
                    set_weight_act_quantize_params(module)
                else:
                    recon_model(module)

        # Start calibration
        recon_model(qnn)

        if args.act_quant and args.order == "after" and args.waq is False:
            """Case 1"""
            set_act_quantize_params(
                qnn, cali_data=cali_data, awq=args.awq, order=args.order
            )

        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn


"""calib data generation"""
def random_calib_data_generator(
    args, num_samples, device, t_mode, num_timesteps
):
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    
    batch = next(iter(loader))
    x, x_lengths = batch['x'], batch['x_lengths']
    y, y_lengths = batch['y'], batch['y_lengths']
    spk = batch['spk']
    # random the mel
    shape = y.shape
    
    # calib_data = calib_data.to(device)
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = torch.randn(*shape, device=device)
    y_lengths = y_lengths.to(device)
    spk = spk.to(device)
    t = generate_t(args, t_mode, num_samples, num_timesteps, device)
    return x, x_lengths, y, y_lengths, t, spk


def raw_calib_data_generator(
    args, num_samples, device, t_mode, num_timesteps
):
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    
    batch = next(iter(loader))
    x, x_lengths = batch['x'], batch['x_lengths']
    y, y_lengths = batch['y'], batch['y_lengths']
    spk = batch['spk']
    # calib_data = calib_data.to(device)
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = y.to(device)
    y_lengths = y_lengths.to(device)
    spk = spk.to(device)
    t = generate_t(args,t_mode, num_samples, num_timesteps, device)
    return x, x_lengths, y, y_lengths, t, spk


def forward_t_calib_data_generator(
    args, num_samples, device, t_mode, num_timesteps, model
):
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    
    batch = next(iter(loader))
    x, x_lengths = batch['x'], batch['x_lengths']
    y, y_lengths = batch['y'], batch['y_lengths']
    spk = batch['spk']
    # calib_data = calib_data.to(device)
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = y.to(device)
    y_lengths = y_lengths.to(device)
    spk = spk.to(device)
    t = generate_t(args,t_mode, num_samples, num_timesteps, device)
    y_t = model.get_x_t(x, x_lengths, y, y_lengths, t, spk)
    
    return x, x_lengths, y_t, y_lengths, t, spk



def backward_t_calib_data_generator(
    args, num_samples, device, t_mode, num_timesteps, model
):
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    
    batch = next(iter(loader))
    x, x_lengths = batch['x'], batch['x_lengths']
    y, y_lengths = batch['y'], batch['y_lengths']
    spk = batch['spk']
    # calib_data = calib_data.to(device)
    # print('x:',x.shape)
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = y.to(device)
    y_lengths = y_lengths.to(device)
    spk = spk.to(device)
    t = generate_t(args,t_mode, num_samples, num_timesteps, device)
    _, y_t, y_lengths,_ = model.forward_calib_backward(x, x_lengths, num_timesteps,t, temperature=1.5, stoc=False, spk=spk, length_scale=1.0)
    
    return x, x_lengths, y_t, y_lengths, t, spk



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    
    
    # quant params
    parser.add_argument("--data_dir", type=str, help="ImageNet dir")

    parser.add_argument(
        "--seed", default=3, type=int, help="random seed for results reproduction"
    )

    # quantization parameters
    parser.add_argument(
        "--n_bits_w", default=4, type=int, help="bitwidth for weight quantization"
    )
    parser.add_argument(
        "--channel_wise",
        action="store_true",
        help="apply channel_wise quantization for weights",
    )
    parser.add_argument(
        "--n_bits_a", default=4, type=int, help="bitwidth for activation quantization"
    )
    parser.add_argument(
        "--act_quant", action="store_true", help="apply activation quantization"
    )
    parser.add_argument("--disable_8bit_head_stem", action="store_true")

    # weight calibration parameters
    parser.add_argument(
        "--calib_num_samples",
        default=1024,
        type=int,
        help="size of the calibration dataset",
    )
    parser.add_argument(
        "--iters_w", default=20000, type=int, help="number of iteration for adaround"
    )
    parser.add_argument(
        "--weight",
        default=0.01,
        type=float,
        help="weight of rounding cost vs the reconstruction loss.",
    )
    parser.add_argument(
        "--keep_cpu", action="store_true", help="keep the calibration data on cpu"
    )

    parser.add_argument(
        "--wwq",
        action="store_true",
        help="weight_quant for input in weight reconstruction",
    )
    parser.add_argument(
        "--waq",
        action="store_true",
        help="act_quant for input in weight reconstruction",
    )

    parser.add_argument(
        "--b_start",
        default=20,
        type=int,
        help="temperature at the beginning of calibration",
    )
    parser.add_argument(
        "--b_end", default=2, type=int, help="temperature at the end of calibration"
    )
    parser.add_argument(
        "--warmup",
        default=0.2,
        type=float,
        help="in the warmup period no regularization is applied",
    )

    # activation calibration parameters
    parser.add_argument("--lr", default=4e-5, type=float, help="learning rate for LSQ")

    parser.add_argument(
        "--awq",
        action="store_true",
        help="weight_quant for input in activation reconstruction",
    )
    parser.add_argument(
        "--aaq",
        action="store_true",
        help="act_quant for input in activation reconstruction",
    )

    parser.add_argument(
        "--init_wmode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for weight",
    )
    parser.add_argument(
        "--init_amode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for activation",
    )
    # order parameters
    parser.add_argument(
        "--order",
        default="before",
        type=str,
        choices=["before", "after", "together"],
        help="order about activation compare to weight",
    )
    parser.add_argument("--prob", default=1.0, type=float)
    parser.add_argument("--input_prob", default=1.0, type=float)
    parser.add_argument("--use_adaround", action="store_true")
    parser.add_argument(
        "--calib_im_mode",
        default="random",
        type=str,
        choices=["random", "raw", "raw_forward_t", "noise_backward_t"],
    )
    parser.add_argument(
        "--calib_t_mode",
        default="random",
        type=str,
        choices=["random", "1", "-1", "mean", "uniform" , 'manual' ,'normal' ,'poisson'],
    )
    parser.add_argument(
        "--calib_t_mode_normal_mean",
        default=0.5,
        type=float,
        help='for adjusting the weights in the normal distribution'
    )
    parser.add_argument(
        "--calib_t_mode_normal_std",
        default=0.35,
        type=float,
        help='for adjusting the weights in the normal distribution'
    )
    parser.add_argument("--out_path", default="", type=str)
    args = parser.parse_args()
    
    
    
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    generator.cuda().eval()
    
    # quant model
    generator = quant_model(args, generator, 100)
    
    # print(f'Number of parameters: {generator.model.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.model.forward_infer(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_{i}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')




