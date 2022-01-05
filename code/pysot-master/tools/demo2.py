from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
pip install opencv-python
import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from tkinter import *
from tkinter import filedialog, messagebox
import tkinter.font as tkFont


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        # cap = cv2.VideoCapture(0)
        # # warmup
        # for i in range(5):
        #     cap.read()
        # while True:
        #     ret, frame = cap.read()
        #     if ret:
        #         yield frame
        #     else:
        #         break
        messagebox.showinfo("Hint", "Please select a video file or picture folder")
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def get_frames1(video_name):
    if not video_name:
        messagebox.showinfo("Hint", "Please select a video file or picture folder")
        return 0
    elif video_name.endswith('avi') or video_name.endswith('mp4') or video_name.endswith('MOV'):
        return 1
    elif  os.path.isdir(video_name):
        return 1

    else:  messagebox.showinfo("Hint", "Please select a video file or picture folder")
    return 0





def main():
    def selectPath():
        # path_=askdirectory()
        path_ = filedialog.askopenfilename()
        #path_ = filedialog.askdirectory()
        path.set(path_)
        # print(path.get())

    def startTrack():
        # load config
        cfg.merge_from_file(args.config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(args.snapshot,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        tracker = build_tracker(model)

        first_frame = True

        args.video_name = path.get()

        if args.video_name:
            video_name = args.video_name.split('/')[-1].split('.')[0]
        else:
            video_name = 'webcam'


        flag=get_frames1(args.video_name)
        if flag==0:
            return

        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        for frame in get_frames(args.video_name):
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                cv2.imshow(video_name, frame)
                cv2.waitKey(40)

    top = Tk()
    path = StringVar()
    top.title("Soccer Player Tracking System")
    ft = tkFont.Font(family='Fixdsys', size=20, weight=tkFont.BOLD)
    mycolor = "#F8F8FF"
    top.geometry("600x300")
    top.configure(bg=mycolor)

    Label(top,text=" Soccer Player Tracking System",font=ft,pady=50,bg="#F8F8FF").grid(row=0, column=1)
    Label(top, text="Video path").grid(row=1, column=0,padx=50)
    Entry(top, textvariable=path,relief="groove",width=30).grid(row=1, column=1)
    Button(top, text="Path Selection", command=selectPath).grid(row=1, column=2,padx=50,ipadx=30)
    Button(top, text="Select tracking target", command=startTrack).grid(row=2, column=1,pady=50,ipadx=50)

    top.mainloop()


    # # load config
    # cfg.merge_from_file(args.config)
    # cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    # device = torch.device('cuda' if cfg.CUDA else 'cpu')
    #
    # # create model
    # model = ModelBuilder()
    #
    # # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)
    #
    # # build tracker
    # tracker = build_tracker(model)
    #
    #
    #
    # first_frame = True
    # if args.video_name:
    #     video_name = args.video_name.split('/')[-1].split('.')[0]
    # else:
    #     video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    # for frame in get_frames(args.video_name):
    #     if first_frame:
    #         try:
    #             init_rect = cv2.selectROI(video_name, frame, False, False)
    #         except:
    #             exit()
    #         tracker.init(frame, init_rect)
    #         first_frame = False
    #     else:
    #         outputs = tracker.track(frame)
    #         if 'polygon' in outputs:
    #             polygon = np.array(outputs['polygon']).astype(np.int32)
    #             cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
    #                           True, (0, 255, 0), 3)
    #             mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
    #             mask = mask.astype(np.uint8)
    #             mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
    #             frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
    #         else:
    #             bbox = list(map(int, outputs['bbox']))
    #             cv2.rectangle(frame, (bbox[0], bbox[1]),
    #                           (bbox[0]+bbox[2], bbox[1]+bbox[3]),
    #                           (0, 255, 0), 3)
    #         cv2.imshow(video_name, frame)
    #         cv2.waitKey(40)


if __name__ == '__main__':
    main()
