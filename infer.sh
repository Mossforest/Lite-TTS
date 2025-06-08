export PYTHONPATH=.
# python inference.py \
#     -f /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/guanwenhao-240108090032/Speech-Backbones-main/Grad-TTS/resources/filelists/synthesis.txt \
#     -c /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/guanwenhao-240108090032/Speech-Backbones-main/Grad-TTS/checkpts/grad-tts-libri-tts.pt \
#     -t 100 \
#     -s 0



python ptq4dm.py \
    -f /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/guanwenhao-240108090032/Speech-Backbones-main/Grad-TTS/resources/filelists/synthesis.txt \
    -c /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/guanwenhao-240108090032/Speech-Backbones-main/Grad-TTS/checkpts/grad-tts-libri-tts.pt \
    -t 1000 \
    -s 0 \
    --n_bits_w 4 --channel_wise --n_bits_a 8  --act_quant --order together --wwq --waq --awq --aaq \
    --weight 0.01 --input_prob 0.5 --prob 0.5 --iters_w 100 --calib_num_samples 16 \
    --calib_im_mode raw_forward_t \
    --calib_t_mode 1 --calib_t_mode_normal_mean 0.4 --calib_t_mode_normal_std 0.4


