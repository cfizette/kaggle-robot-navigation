#!/usr/bin/env bash

mkdir -p data
cd data
kaggle competitions download -c career-con-2019
unzip *.zip
chmod 777 *.csv