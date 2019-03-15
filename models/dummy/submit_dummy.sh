#!/usr/bin/env bash

python predict_dummy.py

kaggle competitions submit -c career-con-2019 -f submission.csv -m "Dummy predictions"