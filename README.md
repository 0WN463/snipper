# Frame Snipper

A Python program that trims off undesired of clips that matches in visual similarity with the undesired templates

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Quick Start

Example: `python snip.py -i input_folder -o output_folder template_file.csv` 

If no input folder is specified, it will default to using `input` as the the folder.

If no output folder is specified, it will default to using `output` as the the folder.

If you do not wish to trim the clip from the template file, add the `-s` or `--skip-templates` option.

Where the content of the template_file.csv is in the form of

```
FILE_NAME, START_TIMESTAMP, END_TIMESTAMP, START_RANGE, END_RANGE
```

Where FILE_NAME specifies the video file to act as the template

START_TIMESTAMP/END_TIMESTAMP is the start/end of the segment to excise

START_RANGE/END_RANGE is the region where the program should search.

Timestamps can be specified in seconds or MM:SS.

Note that negative indexing is allowed for the time stamps

For example

```
Ep1.mp4,1:49,3:08,0,300
Ep1.mp4,22:29,23:35,-300,0
```

To check if the timestamps specified is correct, the `-f` or `--viewframe` argument can be added to the program.

This will display the start/end frames as specified in the template

usage: snip.py [-h] [-i I] [-o O] [-f] [-s] templates


## Example Workflow

Here, we have a set of clips from Higurashi No Naku Koroni. 
Each clip has the same opening and ending sequence that we wish to excise.
In this example, we will be splicing the clips based on the template on Episode 6.

Ideally, the starting/ending frame of the segment which we want to excise remain relatively unchange for a period of time, or else the program may be unable to find this frame in the target clip (unless the granularity is set to 0). 
Thus, ideally, our clip should start on a frame that does not rapidly change, ie it does not cut to a different scene.

As we see in the starting sequence below, the scene does not change for a decent period of time. 
For good performance, the scene should remain unchanged for at least 1 second so that the program can correctly identify the frame.

## Authors

* **0WN3D** - *Initial work* - [0WN463](https://github.com/0WN463)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

