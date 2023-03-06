# Hello!

The project is an attempt to improve on the examples given by Taichi, make them more interesting, and use them to show the classic results of computer graphics.

## Prerequisites

- Python: 3.7/3.8/3.9/3.10 (64-bit)
- OS: Windows, OS X, and Linux (64-bit)

## Installation

Taichi is available as a PyPI package:

```bash
pip install taichi
```

## Items

### pbf2d_game.py:

![image](https://github.com/NextoneX/Graphics_by_Taichi/blob/main/resource/demo.gif)

​	A simple simulation of `Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.`

​	I added more interactivity to it, now you can create new particle, generate attraction, control gravity and change parameters.

​	If you'd like to experience more features, please leave me a message.  

#### Key description:

- Left Mouse Button : create new particle;
- Right Mouse Button: generate attraction;
- R: restart;  
- Space: Stop board;  
- WSAD/arrow keys: control gravity;

#### Adjustable parameters:

- `Initial particle number` : The modification takes effect upon restart
- `New Particles` : The number of new particles produced at a time. The modification takes effect immediately
- `Attraction Strength` : The magnitude of attraction ( caused by Right Mouse Button ). The modification takes effect immediately
- `Board Period` & `Board Range` : Control the movement of board. The modification takes effect after the board has completed a full cycle of movement





