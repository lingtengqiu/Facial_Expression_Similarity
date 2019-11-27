#--exclude=SH-IDC1-10-5-30-150
srun -k --exclude=SH-IDC1-10-5-30-[150,39]  -p MediaA -n1 --gres gpu:4 --ntasks-per-node=1 -J FEC \
python tools/train.py \
        --config-file ./configs/e2e_fec_net.yaml
2>&1 | tee train_5.log
