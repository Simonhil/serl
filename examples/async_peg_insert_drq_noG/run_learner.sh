export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200\
    --critic_actor_ratio 4 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/shilber/delete/good_examples/hex_random_reset.pkl\
    --checkpoint_period 2500 \
    --checkpoint_path /home/shilber/delete/test1\