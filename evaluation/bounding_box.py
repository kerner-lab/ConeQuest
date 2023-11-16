import numpy as np


def mask_to_boxes(mask):

    # -> Array['N, 4', int]:

    """ Convert a boolean (Height x Width) mask into a (N x 4) array of NON-OVERLAPPING bounding boxes
    surrounding "islands of truth" in the mask.  Boxes indicate the (Left, Top, Right, Bottom) bounds
    of each island, with Right and Bottom being NON-INCLUSIVE (ie they point to the indices AFTER the island).

    This algorithm (Downright Boxing) does not necessarily put separate connected components into
    separate boxes.

    You can "cut out" the island-masks with
        boxes = mask_to_boxes(mask)
        island_masks = [mask[t:b, l:r] for l, t, r, b in boxes]
    """

    max_ix = max(s+1 for s in mask.shape)   # Use this to represent background
    # These arrays will be used to carry the "box start" indices down and to the right.
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # Propagate the earliest x-index in each segment to the bottom-right corner of the segment
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i-1, j] if i>0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    # Propagate the earliest y-index in each segment to the bottom-right corner of the segment
    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j-1] if j>0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    # Find the bottom-right corners of each segment
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False)==-1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False)==-1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    # Extract the boxes, getting the top-right corners from the index arrays
    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None]+1, y_stops[:, None]+1])
    return ltrb_boxes