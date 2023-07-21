#!/bin/bash

# Q0
python3 ft.py --task ft   --model bert-tiny,bert-med --dataset amazon --k 1,8,128

# Q1
python3 icl.py --task icl  --model med,full --dataset babi --k 0,1,16
python3 icl.py --task icl  --model med --dataset xsum --k 0,1,4 --prompt none,tldr,custom

# Q2
python3 ft.py --task ft   --model med --mode first,last,middle,lora4,lora16 --dataset xsum,babi --k 0,1,8,64

# Q3
python3 icl.py --task icl  --model med --dataset babi --k 16 --repeats 5
python3 ft.py --task ft   --model med --dataset babi --k 16 --repeats 5 --mode lora16
