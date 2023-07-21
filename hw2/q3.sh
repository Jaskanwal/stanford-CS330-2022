#!/bin/bash
for K in 1 2 4 6 8 10
do
  echo "---protonet, K=$K---" >> hw3_q3.txt
  python3 protonet.py --log_dir ./logs/protonet/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16 --num_way 5 --num_support $K --num_query 10 --test --checkpoint_step 4200 >> hw3_q3.txt
  echo -e "\n" >> hw3_q3.txt
done


for K in 1 2 4 6 8 10
do
  echo "---MAML, K=$K---" >> hw3_q3.txt
    python3 maml.py --log_dir ./logs/maml/omniglot.way_5.support_1.query_15.inner_steps_1.inner_lr_0.4.learn_inner_lrs_True.outer_lr_0.001.batch_size_16 --num_way 5 --num_support $K --num_query 10 --test --checkpoint_step 5200 >> hw3_q3.txt
  echo -e "\n" >> hw3_q3.txt
done
