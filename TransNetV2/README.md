## Inference Usage

-build docker container: docker build -t transnet -f inference/Dockerfile .

-docker run -it --rm --gpus 1 -v /path/to/videos/directory:/tmp transnet transnetv2_predict /tmp/[video_name].mp4 [--visualize] [--makeclips]

### flags
--visualize: generate frame pngs
--makeclips: generate segmented .mp4 clips from scenes.txt frame boundaries

## Pretrained model
-model weights(~/inference/transnetv2-weights/) and dataset stored using git lfs

## Git LFS

-model params stored with git lfs, either download it and pull or just download the zip.
