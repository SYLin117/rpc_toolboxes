from PIL import Image, ImageOps

# open image
img = Image.open("test_image.png")

# border color
color = "green"

# top, right, bottom, left
border = (20, 10, 20, 10)

new_img = ImageOps.expand(img, border=border, fill=color)

# save new image
# new_img.save("test_image_result.jpg")

# show new bordered image in preview
new_img.show()
