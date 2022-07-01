![Soccer Player Tracker](https://github.com/MaxyZhu75/High-Performance-Soccer-Player-Tracking-System-with-CNN/blob/main/summary/ScreenShots9.png)
# :camera: High Performance Soccer Player Tracking System with Convolutional Neural Networks
Visual object tracking has many practical applications in sports, such as tracking motion trajectory, recording player’s speed and helping make strategies. However, modern soccer has become a very fast-paced affair and with players who possess blistering pace and acceleration to volt forward. Considering that the video streams have a higher frame rate, achieving top tracking performance over real-time speed is considered as one of the main challenges. In this paper, we aim to implement a comprehensive soccer player tracking system which can run at frame-rates beyond real-time, and is easy to use in practice.

![GUI](https://github.com/MaxyZhu75/High-Performance-Soccer-Player-Tracking-System-with-CNN/blob/main/summary/GUI.png)

We equipped the tracker with a state-of-the-art Siamese region proposal network (Siamese-RPN), and trained the network end-to-end off-line with Youtube-BB dataset. In the inference phase, the tracking task can be defined as a local one-shot detection task for a seed up. In order to optimize the feature extractor network in the model, we tested and evaluated different backbones on the public dataset VOT2021. Experiments illustrate that ResNet-50 serving as a backbone in the Siamese-RPN tracking model actually outperforms the modified AlexNet on several aspects, including expected average overlap(EAO), accuracy and robustness. A comprehensive soccer player tracking system also need to manage interaction. As shown in Fig. 1, we used the Tkinter for front-end design, which is one of the most popular graphical user interface(GUI) libraries in Python. Finally, by providing a tracking demo, we can demonstrate that our soccer player tracker runs effectively at least 60 FPS frame rate in a soccer game video stream.

## Problem Statement & Way to Approach
Please check out the [Project Paper](https://github.com/MaxyZhu75/High-Performance-Soccer-Player-Tracking-System-with-CNN/blob/main/summary/Paper.pdf) for details.
## Tracking Results
Please check out the [Screenshots](https://github.com/MaxyZhu75/High-Performance-Soccer-Player-Tracking-System-with-CNN/blob/main/summary/ScreenShots6.png) and [Project Paper](https://github.com/MaxyZhu75/High-Performance-Soccer-Player-Tracking-System-with-CNN/blob/main/summary/Paper.pdf) for details.

## Implementation Reference
[1] F. Zhang, Q. Wang, B. Li, Z. Chen, J. Zhou. PysSOT, Github repository, https://github.com/STVIR/pysot, 2018.

## :calling: Contact
Thank you so much for your interests. Note that this project can not straightforward be your course assignment solution. Do not download and submit my code without any change. Feel free to reach me out and I am happy to modify this Soccer Player Tracking System further with you.
* Email: maoqinzhu@umass.edu or zhumaxy@gmail.com
* LinkedIn: [Max Zhu](https://www.linkedin.com/in/maoqin-zhu/)
