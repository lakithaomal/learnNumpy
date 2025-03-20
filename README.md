# learnNumpy

Numpy is the base for multiple other libraries - For example Pandas. 


## EX 1: 
EX1 is based off of this [video](https://www.youtube.com/watch?v=QUT1VHiLmmI&t=864s)


## What is numpy 

It is a multidimensianal array library. 


1d, 2d or upto nd. 

Why not lists. Numpy arrays are much faster than lists. The reson being numpy uses fixed types. 

In a list the int, 5 is represented as a long which is 8 bytes. However in numpy we can go down to int8 which is one byte (default is int32 (4 bytes)). Further a list a lot more information is required.
- Size             - Long (4 bytes)
- Reference Count  - Long (8 bytes)
- Object Type      - Long (8 bytes)
- Object Value     - Long (8 bytes)

Becase numpy is less bytes and the computer reads it much faster. Also numpy douesnt require type checking - and also numpy utilizeds continous memory. Unlike lists  which saves items which are scattered around. 
