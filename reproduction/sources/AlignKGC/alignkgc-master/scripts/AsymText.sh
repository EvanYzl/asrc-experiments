# $1 = /path/to/DBP-5L/  $2 = ea_percent  $3 = ra_percent  $4 - /path/to/mbert/

python -m AlignKGC.alignkgc_base \
    --trainer_module "AlignKGC.Asymmetric_trainer" \
    --trainer_class "AsymText" --multiseed 3 \
    --max_epochs 60 --raloss_coeff 0.8 \
    --dbp5l $1 --ea_percent $2 --ra_percent $3 --mbert_path $4
