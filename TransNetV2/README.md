## Inference Usage

-build docker container: docker build -t transnet -f inference/Dockerfile .

-docker run -it --rm --gpus 1 -v /path/to/videos/directory:/tmp transnet transnetv2_predict /tmp/[video_name].mp4 [--visualize] 

## Pretrained model
-model weights(~/inference/transnetv2-weights/) and dataset stored using git lfs
