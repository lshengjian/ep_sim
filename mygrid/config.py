STAY =  0
UP =    1
RIGHT = 2
DOWN =  3
LEFT =  4

ACT_DIRS={
   LEFT:  (-1,0),
   DOWN:  (0,1),
   RIGHT: (1,0),
   UP:    (0,-1)
}
ACT_NAMES={
   STAY:'STAY',
   LEFT:'LEFT',
   DOWN:'DOWN',
   RIGHT:'RIGHT',
   UP:'UP'
}
COLORS={
   '-': (255,255,255),
   'S': (44,254,105),
   'X': (244,244,95),
   'G': (0,255,255),
}
WINDOWS_SIZE=(640,600)
IN_GOAL=1
IN_FORBIDDEN=-1
OUT_BOUND=-1

MAPS = {
    "1x3": ["-SG"],
    "2x2": ["SX", "-G"],
    "4x4": ["S--S", "-X-X", "---X", "X--G"],
    "8x8": [
        "S------S",
        "--------",
        "---X----",
        "-----X--",
        "--SX----",
        "--X---X-",
        "-X--X-X-",
        "---X---G"
    ]
}

