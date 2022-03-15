import cv2
from PIL import Image, ImageFilter
import numpy as np


def dropShadow(image, offset=(20, 20), background=0xDADADA, shadow=0xA2A2A2,
               border=8, iterations=5, mask=None):
    """
    Add a gaussian blur drop shadow to an image.

    image       - The image to overlay on top of the shadow.
    offset      - Offset of the shadow from the image as an (x,y) tuple.  Can be
                  positive or negative.
    background  - Background colour behind the image.
    shadow      - Shadow colour (darkness).
    border      - Width of the border around the image.  This must be wide
                  enough to account for the blurring of the shadow.
    iterations  - Number of times to apply the filter.  More iterations
                  produce a more blurred shadow, but increase processing time.
    """

    # Create the backdrop image -- a box in the background colour with a
    # shadow on it.
    totalWidth = image.size[0] + abs(offset[0]) + 2 * border
    totalHeight = image.size[1] + abs(offset[1]) + 2 * border
    back = Image.new(image.mode, (totalWidth, totalHeight), background)

    # Place the shadow, taking into account the offset from the image
    shadowLeft = border + max(offset[0], 0)
    shadowTop = border + max(offset[1], 0)
    back.paste(shadow, [shadowLeft, shadowTop, shadowLeft + image.size[0], shadowTop + image.size[1]], mask=mask)

    # # Apply the filter to blur the edges of the shadow.  Since a small kernel
    # # is used, the filter must be applied repeatedly to get a decent blur.
    n = 0
    while n < iterations:
        back = back.filter(ImageFilter.BLUR)
        n += 1
    #
    # Paste the input image onto the shadow backdrop
    imageLeft = border - min(offset[0], 0)
    imageTop = border - min(offset[1], 0)
    back.paste(image, (imageLeft, imageTop),mask=mask)

    return back


if __name__ == "__main__":
    import sys

    # image = Image.open('test_img/obj.jpg')
    # image.thumbnail((200, 200), Image.ANTIALIAS)
    #
    # dropShadow(image).show()
    # # dropShadow(image, background=0xeeeeee, shadow=0x444444, offset=(0, 5)).show()

    obj = Image.open('test_img/obj.jpg')
    mask = Image.open('test_img/mask.png').convert('1')
    where = np.where(np.array(mask))  # value == 255 location
    # where = np.vstack((where[0], where[1]))  ## ian added
    assert len(where[0]) != 0
    assert len(where[1]) != 0
    assert len(where[0]) == len(where[1])
    area = len(where[0])
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)

    obj = obj.crop((x1, y1, x2, y2))
    mask = mask.crop((x1, y1, x2, y2))
    # obj.show()
    dropShadow(obj, mask=mask).show()