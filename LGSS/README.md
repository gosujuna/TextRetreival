Run scene segmentation using this command: python segment_scenes.py --url=<YouTube Video URL> --config=lgss/config/inference.py --threshold=<Threshold used to decide which shot boundaries are scene boundaries>

	For example: python segment_scenes.py --url=https://www.youtube.com/watch?v=TyRVyo5zSYU --config=lgss/config/inference.py --threshold=0.7

 

The original video is downloaded at "./data/<YouTube Video ID>/video/video.mp4" and the segmented scenes are downloaded in the "./data/<YouTube Video ID/scene_video/video" directory as separate .mp4 files