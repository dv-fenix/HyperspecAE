python -u ../src/extract.py\
	-src_dir ../data/Samson/\
	-ckpt ../logs/hyperspecae_final.pt\
	-save_dir ../imgs/\
	-num_bands 156\
	-end_members 3\
	-encoder_type deep\
	-soft_threshold SReLU\
	-activation Leaky-ReLU\
	-gaussian_dropout 0.2\
	-threshold 1.0\