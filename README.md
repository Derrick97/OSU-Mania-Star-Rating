# OSU-Mania-Star-Rating
The entire algorithm is based on [Crz]sunnyxxy and ChlorieHCl's paper. Available here: https://docs.google.com/document/d/1BwbIftjamRm817JKht_U-Yv1W6wqQJjHv_pSj9WtAtc/edit

Note that this is not the prevailing star rating system used in Osu! now. As all we know it has lots of flaws, such as inflating stars for spamming, underrated stars for technical based map and jack map. This project tries to find another way to calculate better star rating.

# Current Status
The current status can be checked on https://github.com/users/Derrick97/projects/1
If you want to join me on this project, please email derrickwolf@outlook.com.

Current Status Summary: Can print metadata, able to read notes from .osu file.

# How to run the program in Command Line:
All code are based on python 3.6.8. Make sure you have at least python 3.4 installed.

For Windows Users:
1. ctrl + r, type "cmd", open the terminal.
2. cd into the directory, type this command into the terminal, replace 'filename' with the path of the .osu file, and press enter.
 ```bash
python algorithm.py ('filename')
```
eg.:
 ```bash
python algorithm.py "C:\Users\oscar\AppData\Local\osu!\Songs\581729 jioyi - cyanine\jioyi - cyanine (Rivals_7) [Ultimate].osu"
```
3. The terminal will display the metadata, and all the hitting objects are already parsed and stored inside datat structures. 
