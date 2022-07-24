#!/bin/bash

M_list=(50000 6000    1200000 100000)
N_list=(50000 6000    6000    100000)
K_list=(50000 1200000 6000    5000)
P_list=(192 384 768 1536 3072)

for ((i = 0; i < 4; i++)); do
    M=${M_list[$i]}
    N=${N_list[$i]}
    K=${K_list[$i]}
    printf "\n\n********** M = $M, N = $N, K = $K **********\n\n"
    for ((j = 0; j < 5; j++)); do
        P=${P_list[$j]}
        ./cosma_statistics -m $M -n $N -k $K -P $P
    done
done
