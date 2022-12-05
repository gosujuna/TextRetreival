from __future__ import print_function

from mmcv import Config
import os
import os.path as osp
import shutil
import argparse

from pytube import YouTube

import sys
sys.path.append('./pre/ShotDetection')
import shotdetection
import time

sys.path.append('./lgss')
import run


def download_video(url, resolution=None):
    yt = YouTube(url)

    videoID = url.split("=")[1]
    os.makedirs("./data/{}".format(videoID), exist_ok = True)
    os.makedirs("./data/{}/video".format(videoID), exist_ok = True)
    video_save_path = "./data/{}/video".format(videoID)
    
    yt.streams.get_highest_resolution().download(video_save_path)
    shutil.move(osp.join(video_save_path,os.listdir(video_save_path)[0]),osp.join(video_save_path,"video.mp4"))
    print("Video ID: {}".format(videoID))
    # print(video_save_path)
    return video_save_path, videoID

def run_detection(video_path):
  save_data_root_path = '/'.join(video_path.split('/')[:3])
  shotdetection.main(video_path=video_path, save_data_root_path=save_data_root_path)


def main(args):
    url = args.url
    cfg = args.config
    threshold = args.threshold
    # resolution = args.resolution
    
    # download the YouTube video
    print("Downloading YouTube video ...")
    video_dir, videoID = download_video(url)
    video_path = video_dir + '/video.mp4'
    # print(video_dir, videoID)
    print("YouTube video downloaded!", '\n')

    # detect the shots
    print("Running Shot Detection ...")
    run_detection(video_path)
    print("Shot Detection complete!", '\n')

    # segment the scenes
    print("Running Scene Segmentation ...")
    run.main(cfg, videoID, threshold)
    print("Scene Segmentation complete!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Download YT video")
  parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=mhDBWiTfNCU', help='URL of YouTube video to perform Scene Segmentation on')
  parser.add_argument('--config', type=str, default='lgss/config/inference.py', help='path to config file')
  parser.add_argument('--threshold', type=float, default=0.8, help='threshold for scene boundary detection')

  args = parser.parse_args()
  main(args)