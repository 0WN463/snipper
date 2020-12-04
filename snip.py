# Import everything needed to edit video clips
from moviepy.editor import *
from PIL import Image
import numpy as np
from time import time
import os
import subprocess

## Timestamps:
## EP1: 34 - 122.5, 21:46 - 23:14
## Ep7: 65

OP_SEARCH_LEN = 300
ED_SEARCH_LEN = 300

def clock(*d_args, **d_kwargs):
    def wrap(func):
        def inner(*args, **kwargs):
            start = time()
            res = func(*args, **kwargs)
            print(d_kwargs['msg'], "took", time() - start)
            return res
        return inner
    return wrap

def ssd(template, target):
    im1, template_weight = template
    im2, image_weight = target
    diff = im1 - im2
    num = np.sum(diff ** 2)
    den = np.sqrt(template_weight + image_weight)
    return num/den

def grey_frame(clip, t):
    im = np.average(np.array(clip.get_frame(t)), axis=2)
    total_weight = np.sum(im**2)
    return im, total_weight

@clock(msg="searching")
def search_clip(clip, template, interval, back_interval=False, fixed_length=False):
    start = time()
    (start_template, start_threshold), (end_template, end_threshold), length = template


    if back_interval:
        interval = int(clip.end - interval), int(clip.end) - 1
        it = range(interval[1], interval[0], -1)
        
    else:
        it = range(interval[0], interval[1])
    for t in it:
        target = grey_frame(clip, t)
        diff = ssd(start_template, target)

        if diff <= start_threshold:
            start_index = t
            break
    else:
        assert False, "Unable to find OP"

    if fixed_length:
        return (start_index, start_index + length)

    for t in np.arange(start_index, interval[1]):
        target = grey_frame(clip, t)
        diff = ssd(end_template, target)

        if diff <= end_threshold:
            return (start_index, t)

    assert False, "Unable to find OP"

@clock(msg="templating")
def get_template(template_clip, indices, interval, back_interval=False):
    start, end = indices    
    start_template = grey_frame(template_clip, start)
    end_template = grey_frame(template_clip, end)

    best_start, second_best_start = float("infinity"), float("infinity")
    best_end, second_best_end = float("infinity"), float("infinity")


    if back_interval:
        interval = template_clip.end - interval, template_clip.end
    for t in np.arange(interval[0], interval[1]):
        if t % 50 == 0:
            print(t)
        target = grey_frame(template_clip, t)
        diff = ssd(start_template, target)
        
        if diff <= best_start:
            second_best_start = best_start
            best_start = diff
        else:
            second_best_start = min(second_best_start, diff)
        
        diff = ssd(end_template, target)
        
        if diff <= best_end:
            second_best_end = best_end
            best_end = diff
        else:
            second_best_end = min(second_best_end, diff)

    start_threshold = best_start + 0.5 * (second_best_start - best_start)
    end_threshold = best_end + 0.5 * (second_best_end - best_end)
    print(best_start, second_best_start)
    return (start_template, start_threshold), (end_template, end_threshold), end-start
import mimetypes
start = time()
input_path = 'input'
output_folder = 'output'

template_file = 'Ep1.mp4'
template_path = os.path.join(input_path, template_file)
target_clips = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) \
                and mimetypes.guess_type(os.path.join(input_path, f))[0].startswith('video')]
print(target_clips)


template_clip = VideoFileClip(template_path)

op_indices = (34, 122.5)
op_interval = (0, OP_SEARCH_LEN)
op_template = get_template(template_clip, op_indices, op_interval)

ed_indices = (21*60 + 46, 23*60+14)
ed_interval = ED_SEARCH_LEN
ed_template = get_template(template_clip, ed_indices, ed_interval, back_interval=True)


try:
    os.mkdir('tmp')
except:
    if not os.path.exists("tmp"):
        print("Unable to create tmp folder. Quiting...")
        quit()

try:
    os.mkdir(output_folder)
except:
    if not os.path.exists(output_folder):
        print("Unable to create output_folder folder. Quiting...")
        quit()

for file in target_clips:
    path = os.path.join(input_path, file)
    target_clip = VideoFileClip(path)
    print("Working on", path)
    
    target_op_indices = search_clip(target_clip, op_template, op_interval, fixed_length=True)
    target_ed_indices = search_clip(target_clip, ed_template, ed_interval, back_interval=True, fixed_length=True)
    print("op id", target_op_indices)
    print("ed id", target_ed_indices)

    os.chdir("tmp")
    
    subprocess.call(f'ffmpeg -i {os.path.join("..", path)} -to {target_op_indices[0]} -y  -c copy -map 0  prologue.mp4', shell=True)
    subprocess.call(f'ffmpeg -i {os.path.join("..", path)} -ss {target_op_indices[1]} -to {int(target_ed_indices[0])} -y  -c copy -map 0  middle.mp4', shell=True)
    subprocess.call(f'ffmpeg -i {os.path.join("..", path)} -ss {target_ed_indices[1]} -y -c copy -map 0  epilogue.mp4', shell=True)

    subprocess.call(f'ffmpeg -i prologue.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y intermediate1.ts', shell=True)
    subprocess.call(f'ffmpeg -i middle.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y intermediate2.ts', shell=True)
    subprocess.call(f'ffmpeg -i epilogue.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y intermediate3.ts', shell=True)

    subprocess.call(f'ffmpeg -y -i "concat:intermediate1.ts|intermediate2.ts|intermediate3.ts" -c copy -bsf:a aac_adtstoasc {os.path.join("..", output_folder, file)}', shell=True)
    os.chdir('..')

print("Took", time() - start)    
