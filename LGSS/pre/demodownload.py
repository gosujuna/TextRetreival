import os
import os.path as osp
import shutil

from pytube import YouTube


def main():
    os.makedirs("../data/demo", exist_ok = True)
    os.makedirs("../data/demo/video", exist_ok = True)
    video_save_path = "../data/demo/video"
    yt = YouTube('https://www.youtube.com/watch?v=mhDBWiTfNCU')
    yt.streams.get_highest_resolution().download(video_save_path)
    shutil.move(osp.join(video_save_path,os.listdir(video_save_path)[0]),osp.join(video_save_path,"demo.mp4"))


if __name__ == '__main__':
    main()
# import os
# import os.path as osp
# import shutil

# from pytube import YouTube


# def main():
#     os.makedirs("../data/demo", exist_ok = True)
#     os.makedirs("../data/demo/video", exist_ok = True)
#     video_save_path = "../data/demo/video"
#     url = "https://www.youtube.com/watch?v=---9CpRcKoU"
#     yt = YouTube(url)
#     # yt = YouTube('https://www.youtube.com/watch?v=rT22nYLaVbo')
#     #yt.streams.get_by_resolution("360p").download(video_save_path)
#     yt.streams.get_highest_resolution().download(video_save_path)
#     shutil.move(osp.join(video_save_path,os.listdir(video_save_path)[0]),osp.join(video_save_path,"demo.mp4"))
#     # url = "https://www.youtube.com/watch?v=---9CpRcKo"
#     # os.makedirs("../data/test", exist_ok = True)
#     # os.makedirs("../data/test/video", exist_ok = True)
#     # video_save_path = "../data/test/{}".format(''.join(url.split('=')[1:]))
#     #video_id = "G9zN5TTuGO4_000179_000189"
#     #yt = YouTube('https://www.youtube.com/watch?v=' + video_id)
#     # yt = YouTube(url)
#     # yt.streams.get_by_resolution("360p").download(video_save_path)
#     # shutil.move(osp.join(video_save_path,os.listdir(video_save_path)[0]),osp.join(video_save_path,"video.mp4"))

# if __name__ == '__main__':
#     main()
