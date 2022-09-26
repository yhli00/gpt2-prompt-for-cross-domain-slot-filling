echo "job start time: `date`"

tgt_domains="AddToPlaylist RateBook PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="AddToPlaylist"
n_samples=(50)

for tgt_domain in ${tgt_domains[@]}
do
    for n in ${n_samples[@]}
    do
        CUDA_VISIBLE_DEVICES=2 /Work/liyuhang/envs/gpt2_prompt/bin/python trainer.py \
        --do_train \
        --do_test \
        --batch_size 16 \
        --dev_batch_size 2 \
        --num_train_epochs 10 \
        --domain $tgt_domain \
        --nsamples $n \
        --model_dir ./model_dir \
        --log_dir ./log_dir \
        --learning_rate 6.25e-5 \
        --early_stop 100
    done
done

echo "job end time:`date`"