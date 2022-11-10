#!/bin/bash
for (( i=32; i<=104; i++ ))
do
  qdel 610$i[]
  #qdel 27204[$i]
done