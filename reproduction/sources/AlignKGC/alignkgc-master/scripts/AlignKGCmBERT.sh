# $1 = /path/to/DBP-5L/  $2 = ea_percent  $3 = ra_percent  $4 = /path/to/mbert/dir/

python -m AlignKGC.alignkgc_base \
    --trainer_module "AlignKGC.AlignKGCmBERT_trainer" \
    --trainer_class "AlignKGCmBERT" --multiseed 3 \
    --dbp5l $1 --ea_percent $2 --ra_percent $3 --mbert_path $4
