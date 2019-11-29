import numpy as np

def add_image(**kwargs):
    image_info = {
        "id": 1,
        "source": 2,
        "path": 3,
    }
    image_info.update(kwargs)
    print(image_info)


a = dict()
a['zz'] = 4
add_image(**a)