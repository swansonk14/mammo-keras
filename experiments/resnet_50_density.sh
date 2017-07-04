data_path='/raid/scratch/metadata/cohort_nejm_huge_valid.json'
model_architecture='resnet_50'
model_name='resnet_50_density'
image_pipeline='grayscale_256'
label_pipeline='binary'
batch_size=64
examples_per_epoch=20000
examples_per_val=5000
examples_per_eval=2000

python main.py --data_path=$data_path --model_architecture=$model_architecture --model_name=$model_name --image_pipeline=$image_pipeline --label_pipeline=$label_pipeline --batch_size=$batch_size --examples_per_epoch=$examples_per_epoch --examples_per_val=$examples_per_val --examples_per_eval=$examples_per_eval --load_model --debug
