# $1 = /path/to/DBP-5L/  $2 = ea_percent  $3 = ra_percent

python -m AlignKGC.alignkgc_base \
    --trainer_module "AlignKGC.AlignKGC_trainer" \
    --trainer_class "AlignKGC" --multiseed 3 \
    --dbp5l $1 --ea_percent $2 --ra_percent $3
