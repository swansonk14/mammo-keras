data_path='/raid/scratch/metadata/cohort_nejm_huge_valid.json'
model_name='resnet18_density'
pipeline_config_path='pipeline_configs/resnet18_density.yaml'

python main.py --data_path=$data_path --model_name=$model_name --pipeline_config_path=$pipeline_config_path --debug
