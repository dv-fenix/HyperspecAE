python -u ../src/train.py\
	-src_dir ../data/Samson/\
	-num_bands 156\
	-end_members 3\
	-encoder_type deep\
	-soft_threshold SReLU\
	-activation Leaky-ReLU\
	-batch_size 20\
	-learning_rate 1e-3\
	-epochs 250\
	-gaussian_dropout 0.2\
	-threshold 1.0\
	-objective SAD >> ../logs/training_out.txt