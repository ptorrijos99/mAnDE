#!/bin/bash

sed -r "s'[[:blank:]]+'_'g" ./res/params/databases_copy.txt > ./res/params/databases.txt