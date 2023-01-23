#!/bin/bash
qselect -u pablo.torrijos | xargs qdel

qselect -u pablo.torrijos | xargs qdel -W force