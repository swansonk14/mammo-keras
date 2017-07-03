data_path='/raid/scratch/metadata/cohort_nejm_huge_valid.json'
model_name='resnet_18_density'

python main.py --data_path=$data_path --model_name=$model_name --load_model --debug
