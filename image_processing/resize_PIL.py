# using the Python Image Library (PIL) to resize an image
# works with Python27 and Python32
import os

from PIL import Image

image_file = "arrow_down.png"
img_org = Image.open(image_file)
# get the size of the original image
width_org, height_org = img_org.size
# set the resizing factor so the aspect ratio can be retained
# factor > 1.0 increases size
# factor < 1.0 decreases size
factor = 0.4
width = int(width_org * factor)
height = int(height_org * factor)
# best down-sizing filter
img_anti = img_org.resize((width, height), Image.ANTIALIAS)
# split image filename into name and extension
name, ext = os.path.splitext(image_file)
# create a new file name for saving the result
new_image_file = "%s%s%s" % (name, str(factor), ext)
img_anti.save(new_image_file)
print("resized file saved as %s" % new_image_file)
# one way to show the image is to activate
# the default viewer associated with the image type
import webbrowser
webbrowser.open(new_image_file)