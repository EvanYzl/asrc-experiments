# $1 = /path/to/DBP-5L/  $2 = ea_percent  $3 = ra_percent

python -m AlignKGC.alignkgc_base \
    --trainer_module "AlignKGC.KgcUnion_trainer" \
    --trainer_class "KgcUnion" --multiseed 3 \
    --learning_rate 0.7 --max_epochs 60 --raloss_coeff 0.0 \
    --dbp5l $1 --ea_percent $2 --ra_percent $3 \
    --cuda_num 2
