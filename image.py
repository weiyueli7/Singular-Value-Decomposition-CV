from PIL import Image

def export_image(img_arr, name='test.tiff'):
    """
    Given by the startercode, save the image
    """
    Image.fromarray(img_arr.reshape(28,-1)).save(name)
