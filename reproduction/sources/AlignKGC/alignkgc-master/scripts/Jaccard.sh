# $1 = /path/to/DBP-5L/  $2 = ea_percent  $3 = ra_percent

python -m AlignKGC.alignkgc_base \
    --trainer_module "AlignKGC.Jaccard_trainer" \
    --trainer_class "Jaccard" --multiseed 3 \
    --learning_rate 0.7 --max_epochs 60 --raloss_coeff 0.8 \
    --dbp5l $1 --ea_percent $2 --ra_percent $3
