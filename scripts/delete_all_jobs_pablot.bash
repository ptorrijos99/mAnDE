#!/bin/bash
qselect -u pablot | xargs qdel

qselect -u pablot | xargs qdel -W force