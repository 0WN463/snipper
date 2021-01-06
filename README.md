# Frame Snipper

A Python program that trims off undesired of clips that matches in visual similarity with the undesired templates

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- Python 3

### Dependencies

- numpy 
- openCV
- Pillow

### Installing

Run the following from the command line

```
pip install -r requirements.txt
```

## Quick Start

Example: `python snip.py -i input_folder -o output_folder template_file.csv` 

The content of the template_file.csv is in the form of

```
FILE_NAME, START_TIMESTAMP, END_TIMESTAMP, START_RANGE, END_RANGE
```

Where FILE_NAME specifies the video file to act as the template

START_TIMESTAMP/END_TIMESTAMP is the start/end of the segment to excise

START_RANGE/END_RANGE is the region where the program should search.

Timestamps can be specified in seconds or MM:SS.

Note that negative values is allowed for the time stamps, to indicate duration from the end of the clip.

For example

```
Ep1.mp4,1:49,3:08,0,300
Ep1.mp4,22:29,23:35,-300,0
```

If no input folder is specified, it will default to using `input` as the the folder.

If no output folder is specified, it will default to using `output` as the the folder.

If you do not wish to trim the clip from the template file, add the `-s` or `--skip-templates` option.

To check if the timestamps specified is correct, the `-f` or `--viewframe` argument can be added to the program.

This will display the start/end frames as specified in the template

## Example Workflow
Firstly, we procure a set of video clips with a common segment that we wish to excise.

We manually determine the timestamp of the portion that we wish to excise for the first clip.
Use `python snip.py -i INPUT_FOLDER -f TEMPLATE.csv` to preview the starting/ending of the segments that will be excised.

Once we determined that the template is acceptable, use `python snip.py -i INPUT_FOLDER -o OUTPUT_FOLDER TEMPLATE.csv` to start the excising process.

## Details

- The starting/ending frame of the segment which we want to excise remain relatively unchange for a period of time, or else the program may be unable to find this frame in the target clip (unless the granularity is set to 0). 
Thus, ideally, our clip should start on a frame that does not rapidly change, ie it does not cut to a different scene.

- The starting frame that we excise should be unique enough that the program may not excise based on a false positive. 
Thus, it is not advisable to excise based on a complete black/white frame.

- The program determines the similarity using histogram matching of the HSV channels.

- The program assumes that the duration of the excised clips are constant for all the episodes.
Meaning it does not search for similarity of the end of the template.

## Authors

* **0WN3D** - *Initial work* - [0WN463](https://github.com/0WN463)

## License

This project is licensed under the MIT License 

