# WildpathSegment


## Wildpath Vegetation Segmentation

### Download video example:
https://drive.google.com/file/d/1X3qPsSxme49BljdMKtUnL1PDZTSB-oFm/view

install Dependencies:
```
pip3 install -r requirements.txt
```
run:
```
python3 veg_segmentation.py 

--video input video name

--processor default is gpu, for cpu type False.

--output type name of video output, will be saved in avi format.

--json default is exporting json to each frame, with the annotations. type False to abort it.
```
###### Serializing predicted frames to json file, will provide the object number and it's coordinates (x,y) in the segmented image.
run:
```
python3 veg_segmentation.py --video video_name --processor True --output video_output_name --json True
```
