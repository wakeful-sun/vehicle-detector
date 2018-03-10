import numpy as np


def insert_separator(items, separator):
    row = []
    for index, item in enumerate(items):
        if index:
            row.append(separator)
        row.append(item)
    return row


def create_composite_image(rgb_images, h_span=5, v_span=5, n_columns=3):
    image_shapes = set(map(lambda x: x.shape, rgb_images))
    images_have_same_shape = len(image_shapes) == 1
    if not images_have_same_shape:
        raise NotImplementedError("Can't merge images of different shape")

    h, w, d = image_shapes.pop()
    h_separator_shape = h_span, (w + v_span) * n_columns - h_span, d
    v_separator_shape = w, v_span, d

    h_separator = np.zeros(h_separator_shape, dtype=np.uint8)
    v_separator = np.zeros(v_separator_shape, dtype=np.uint8)

    row_counter = 0
    get_row_images = lambda x: rgb_images[(row_counter * n_columns):(row_counter * n_columns + n_columns)]

    row_images = get_row_images(rgb_images)
    rows = []
    while len(row_images):
        diff = n_columns - len(row_images)
        if diff > 0:
            empty_image = np.zeros((w, h, d), dtype=np.uint8)
            [row_images.append(empty_image) for i in range(diff)]

        row = insert_separator(row_images, v_separator)
        rows.append(np.hstack(row))

        row_counter = row_counter + 1
        row_images = get_row_images(rgb_images)

    image_rows = insert_separator(rows, h_separator)
    resulting_image = np.vstack(image_rows)
    return resulting_image