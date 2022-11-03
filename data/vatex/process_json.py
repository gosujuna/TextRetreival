import csv
import json

filename = "vatex_validation.json"
vatex_json = json.load(open(filename))

vatex_csv =  open(filename.split('.')[0] + ".csv", "w")

vatex_csv.write('"Video ID", "start_time_stamp", "end_time_stamp", "caption"')
vatex_csv.write('\n')

for line in vatex_json:
    i = 0
    num_captions = len(line["enCap"])
    for j in range(num_captions):
        videoID_info = line["videoID"].split("_")
        if len(videoID_info) == 3:
            videoID, start, end = videoID_info
        else:
            start = videoID_info[-2]
            end = videoID_info[-1]
            videoID = videoID_info[:-2]
        vatex_csv.write('{}, {}, {}, {}'.format(videoID, start, end, line["enCap"][j]))
        vatex_csv.write('\n')
vatex_csv.close()