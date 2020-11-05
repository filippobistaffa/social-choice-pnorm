#!/bin/bash

distr="distributions"

for d in $distr/*
do
    for p in $d/p_*
    do
        l="$d/l_${p##*_}"
        c="$d/c_${p##*_}"
        python3 multi_norm.py -u -p $p -l $l -o $c
    done
done
