export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 5\
    --critic_actor_ratio 4 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/shilber/delete/peg_insert__demos_2024-10-18_09-42-36.pkl\
    --checkpoint_period 10000 \
    --checkpoint_path /home/shilber/delete/test1\