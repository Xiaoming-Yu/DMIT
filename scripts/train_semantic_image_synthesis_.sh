GPU_ID=0
DISPLAY_ID=$((GPU_ID*5+7))
python ./train.py \
  --dataroot  ./datasets/birds \
  --checkpoints_dir ./checkpoints \
  --exp_name dmit_semantic_image_synthesis \
  --model_name semantic_image_synthesis \
  --gpu ${GPU_ID} \
  --display_id ${DISPLAY_ID} \
  --display_port 8033 \
  --save_epoch_freq 25 \
  --niter 100 \
  --niter_decay 100 \
  --load_size 143 \
  --fine_size 128 \
  --n_attribute 256 \
  --n_style 8 \
  --batch_size 8 \
  --is_flip \
  --n_image_disblocks  4




  
