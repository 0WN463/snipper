# Import everything needed to edit video clips
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import subprocess
import mimetypes
import argparse
import time
import re

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
channels = [0, 1]

compare_method = cv2.HISTCMP_BHATTACHARYYA

def clock(*d_args, **d_kwargs):
    def wrap(func):
        def inner(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(d_kwargs['msg'], "took", time.time() - start)
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

class VideoFileClip:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)/self.cap.get(cv2.CAP_PROP_FPS))
        self.path = path
    def get_frame(self, t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, round(t*1000))
        ret, frame = self.cap.read()

        if not ret:
            print(ret, t)
        return frame

def grey_frame(clip, t):
    frame = clip.get_frame(t)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_weight = np.sum(grey**2)
    return grey, total_weight

@clock(msg="searching")
def search_clip(clip, template, interval, back_interval=False, fixed_length=False):
    start = time.time()
    (start_hist, start_threshold), (end_hist, end_threshold), length = template
    granularity = 0.5

    if back_interval:
        interval = int(clip.end + interval[0]), int(clip.end + interval[1]) - 1
#        it = np.arange(interval[1]-length+1, interval[0], -granularity)
        it = np.arange(interval[0], interval[1], granularity)
    else:
        it = np.arange(interval[0], interval[1], granularity)
    best = (float("infinity"), None)
    for t in it:
        target = cv2.cvtColor(clip.get_frame(t), cv2.COLOR_BGR2HSV)
        target_hist = cv2.calcHist([target], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        diff = cv2.compareHist(start_hist, target_hist, compare_method)

        if diff < best[0]:
            best = (diff, t)
            
        if diff <= start_threshold:
            start_index = t
            break
    else:
        print("Found best at ", best)
        assert False, "Unable to find start"

    if fixed_length:
        return (start_index, start_index + length)

    for t in np.arange(start_index, interval[1]):
        target = cv2.cvtColor(clip.get_frame(t), cv2.COLOR_BGR2HSV)
        target_hist = cv2.calcHist([target], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        diff = cv2.compareHist(end_hist, target_hist, compare_method)

        if diff <= end_threshold:
            return (start_index, t)

    assert False, "Unable to find end"
def make_hist(template_clip, indices):
    start, end = indices    

    start_template = cv2.cvtColor(template_clip.get_frame(start), cv2.COLOR_BGR2HSV)
    end_template = cv2.cvtColor(template_clip.get_frame(end), cv2.COLOR_BGR2HSV)

    start_hist = cv2.calcHist([start_template], channels, None, histSize, ranges, accumulate=False)
    end_hist = cv2.calcHist([end_template], channels, None, histSize, ranges, accumulate=False)

    cv2.normalize(start_hist, start_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(end_hist, end_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return start_hist, end_hist

@clock(msg="templating")
def get_template(template_clip, indices, interval, back_interval=False):
    start, end = indices    
    start_hist, end_hist = make_hist(template_clip, indices)
    
    best_start, second_best_start = float("infinity"), float("infinity")
    best_end, second_best_end = float("infinity"), float("infinity")

    if back_interval:
        interval = template_clip.end + interval[0], template_clip.end + interval[1]
    for t in np.arange(interval[0], interval[1]):
        target = cv2.cvtColor(template_clip.get_frame(t), cv2.COLOR_BGR2HSV)
        target_hist = cv2.calcHist([target], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        diff = cv2.compareHist(start_hist, target_hist, compare_method)

        if diff <= best_start:
            second_best_start = best_start
            best_start = diff
        else:
            second_best_start = min(second_best_start, diff)

        diff = cv2.compareHist(end_hist, target_hist, compare_method)

        if diff <= best_end:
            second_best_end = best_end
            best_end = diff
        else:
            second_best_end = min(second_best_end, diff)

    start_threshold = best_start + 0.8 * (second_best_start - best_start)
    end_threshold = best_end + 0.8 * (second_best_end - best_end)
    print(best_start, second_best_start)
    return (start_hist, start_threshold), (end_hist, end_threshold), end-start

class Template:
    def __init__(self, file_name, start, end, start_interval, end_interval):
        to_time = lambda time_str: float(time_str) if ":" not in time_str else float(int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1]))
        
        start = to_time(start)
        end = to_time(end)
        print(start, end)
        start_interval = float(start_interval)
        end_interval = float(end_interval)


        assert end > start
        assert end_interval > start_interval

        self.file_name = file_name
        self.indices = (start, end)
        self.interval = (start_interval, end_interval)

        self.is_backwards = start_interval < 0

    def __repr__(self):
        return f"File: {self.file_name}, Index: {self.indices}, Interval: {self.interval}, IsBackwards: {self.is_backwards}"

    def show_frame(self):
        template_clip = VideoFileClip(os.path.join(self.directory, self.file_name))
        start, end = self.indices
        start_frame, _ = grey_frame(template_clip, start)
        end_frame, _ = grey_frame(template_clip, end)

        def draw_time_stamp(img, time_stamp):            
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text((0, 0),time_stamp, 255,font=font)
            draw.text((0, 100),time_stamp, 0,font=font)
        
        im1 = Image.fromarray(start_frame)
        im2 = Image.fromarray(end_frame)

        draw_time_stamp(im1, time.strftime('%H:%M:%S', time.gmtime(start)))
        draw_time_stamp(im2, time.strftime('%H:%M:%S', time.gmtime(end)))
        
        dst = Image.new('L', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0,0))
        dst.paste(im2, (0,im1.height))
        dst.show()
    
parser = argparse.ArgumentParser()
parser.add_argument("templates")
parser.add_argument('-i')
parser.add_argument('-o')
parser.add_argument('-f', '--viewframe', action='store_true', 
    help="shows the frames for each interval")
parser.add_argument('-s', '--skip-templates', action='store_true', 
    help="will retain the clips in the interval for the template files")
args = parser.parse_args()

templates = []

try:
    with open(args.templates, "r") as f:
        for line in f:
            if not re.match(r"^.+,([\d\.]+|\d+:\d+),([\d\.]+|\d+:\d+),-?[\d\.]+,-?[\d\.]+$", line):
                print(line, "is malformed")
                quit()
            else:
                templates.append(Template(*line.split(",")))
except Exception as e:
    print("Cannot open", args.templates)
    print(e)
    quit()
print(templates)
start = time.time()
input_path = args.i if args.i else 'input'
output_folder = args.o if args.o else 'output'


print(input_path)
print(output_folder)

if not os.path.exists(input_path):
    print("Unable to find folder", input_path)
    quit()
if not os.path.exists(output_folder):
    print("Unable to find folder", output_folder)
    quit()

for template in templates:
    template.directory = input_path

if args.viewframe:
    for template in templates:
        template.show_frame()

    quit()
    
template_file = templates[0].file_name
template_path = os.path.join(input_path, template_file)
target_clips = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) \
        and mimetypes.guess_type(os.path.join(input_path, f))[0].startswith('video')]
print(target_clips)
print([template.file_name for template in templates])

template_clip = VideoFileClip(template_path)

op_indices = templates[0].indices
op_interval = templates[0].interval
ed_indices = templates[1].indices
ed_interval = templates[1].interval

op_template = get_template(template_clip, op_indices, op_interval)
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

def multi_call(cmds):
    for cmd in cmds:
        p = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        print("RUNNING", cmd)
        if p.returncode != 0:
            print(p.stderr)
            return

for file in target_clips:
    print(file)
    if file in [template.file_name for template in templates] and args.skip_templates:
        print("template found. Skipping...")
        continue
    
    path = os.path.join(input_path, file)
    target_clip = VideoFileClip(path)
    print("Working on", path)
    indices = []
    try:
        indices.append(search_clip(target_clip, op_template, op_interval, fixed_length=True))
    except AssertionError as e:
        pass
    try:
        indices.append(search_clip(target_clip, ed_template, ed_interval, back_interval=True, fixed_length=True))
    except AssertionError as e:
        pass

    if len(indices) == 0:
        continue
    prev = 0


    cmds = []

    for i, (start, end) in enumerate(indices):
        if start == prev:
            start += 0.1

        if prev == 0:
            cmds.append(f'ffmpeg -i {os.path.join("..", path)} -to {start} -y  -c copy -map 0  part{i}.mp4')
        else:
            cmds.append(f'ffmpeg -i {os.path.join("..", path)} -ss {prev} -to {start} -y  -c copy -map 0  part{i}.mp4')
        prev = end

    cmds.append(f'ffmpeg -i {os.path.join("..", path)} -ss {prev} -y  -c copy -map 0  part{i+1}.mp4')

    num_parts = len(cmds)
    for i in range(num_parts):
        cmds.append(f'ffmpeg -i part{i}.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y intermediate{i}.ts')


    pipes = '|'.join([f'intermediate{i}.ts' for i in range(num_parts)])

    cmds.append(f'ffmpeg -y -i "concat:{pipes}" -c copy -bsf:a aac_adtstoasc {os.path.join("..", output_folder, file)}')
    
    print(indices)
    print(cmds)
    os.chdir("tmp")

    multi_call(cmds)
    os.chdir('..')

print("Took", time.time() - start)    
