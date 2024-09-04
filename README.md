# ImageViewer

OpenCV for Python under Windows can not zoom into images displayed with cv2.imshow().
Here is a Python module to display images that you can zoom in via mouse.
It also allows to do windowing, i.e. re-mapping of brightness values to a smaller or wider range.
Windowing is especially useful with images where the channels have more than 8 bit resolution,
but it can also be used to threshold the viewed image.

## Usage

Start it with
```
python ImageViewer.py example.png
```
or import cv_imshow() as a quick replacement for cv2.imshow()
```
from ImageViewer import cv_imshow
```
In both cases you get some hints via console output like:

```
Mouse Usage:
============

Left mouse does everything important:
    * click zooms in/out using default zoom value 1
    * click on border resizes window to remove border
    If zoomed out:
    * drag does zoom via rectangle and updates default zoom value 1
    If zoomed in:
    * drag does move the view

Right mouse does the same, except it never drag moves:
    * click zooms in/out using default zoom value 2
    * click on border resizes window to remove border
    * drag does zoom via rectangle and updates default zoom value 2

Middle mouse:
    * drag does windowing; shift-drag for fine-tuning
    * click resets windowing
    * control-drag zooms in/out around start point depending on vertical position

Mouse wheel:
    zooms in/out at mouse position
```
