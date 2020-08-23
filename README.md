# Video-filters-with-Mask-RCNN-integration 

Extension of Matterport's version of Keras Mask-RCNN to create various video filters

## Requirements

Apart from the usual packages, to run Filters.py you will need:
1) argparse
2) Moviepy

## Effects

With Filters.py you can apply various effects to your video:  
grey_back, blur_back, bright_object, sharp_object, sharp_back, cartoon_back, warm_front_cold_black, edge_person, pencil_sketch_back  

Demonstrated below are gifs of some of the few effects you can recreate with this code:  

|**Demo 1**|**Demo 2**|
| :--: | :--: |
```
|![](demo/fujing.gif)|![](demo/nikki.gif)|
```


## Usage

As an exammple, in terminal enter the following:
```
python Filters.py --input input.mp4 --output output.mp4 --filter cartoon_back
```
Before running, please download the mask_rcnn_coco.h5 file from the [released page](https://github.com/matterport/Mask_RCNN/releases) and put it in the main folder.  

Executed on tensorflow 2.1.0 and keras 2.3.1
