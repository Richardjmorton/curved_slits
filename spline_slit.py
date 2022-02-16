import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates, spline_filter


class PointPicker:
    """
    Input.

    image - image for selecting points on
    """

    def __init__(self, image):

        fig, ax = plt.subplots()
        ax.set_title('Click to select points. Close window to end.')
        im = ax.imshow(image, origin='lower')
        line, = ax.plot([0], [0], '+r')

        self.image = im
        self.line = line
        self.xs = []
        self.ys = []
        self.cid = im.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event.xdata, event.ydata)
        if event.inaxes != self.image.axes:
            return
        self.xs.append(np.round(event.xdata))
        self.ys.append(np.round(event.ydata))
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def return_points(self):
        return np.array((self.xs, self.ys))

    def end_pick(self):
        self.image.figure.canvas.mpl_disconnect(self.cid)
        print('el fin')


def get_spline_points(gpoints: np.ndarray, spacing: int = 1,
                      diff_tol: float = 1e-2) -> np.ndarray:
    """
    Return points along 2Dspline curve (uses 3rd Order).

    Oversampling works by setting the spacing to be smaller to make the
    point density larger. Then by summing the lengths between the
    points together to make up the original desired length the correct
    point spacing is achieved while reducing the impact of any error in
    the splining routine.

    The check for when the inter-point distance (from oversampled
    curve) has reached the desired spacing is achieved by finding the
    cumulative distance along the (oversampled) curve. Then by dividing
    this cumulative distance vector by the original sampling it is
    apparent that every time the desired original spacing is found the
    floor of the cumulative distance will increase by 1.

    Parameters
    ----------
    gpoints - selected guide points along which to calculate spline
              (should be more than 4)

    spacing - Euclidean distance between spline points (approximate
              if oversample not used)

    oversample - set to oversample spline - ensures accurate spacing

    diff_tol - accuracy of spacing. Setting this value smaller leads to longer
               runs times.

    Return
    -------
    Vector of points along curve
    """
    xvec = gpoints[:, 0]
    yvec = gpoints[:, 1]

    max_diff = diff_tol + 1
    up_sample = 2

    # spl = splrep(xvec, yvec)
    tck, u = splprep([xvec, yvec], s=0)
    ospacing = spacing

    while max_diff > diff_tol:

        spacing = np.float64(ospacing) / (10.**up_sample)

        # xnew_l = np.linspace(xvec[0], xvec[-1], int(1 / spacing))
        u = np.linspace(0, 1, int(1 / spacing))

        # ynew_l = splev(xnew_l, spl)
        xnew_l, ynew_l = splev(u, tck)
        point_num = xnew_l.shape[0]

        # linear distance between spline points
        diff_vec_l = np.sqrt((np.roll(xnew_l, -1) - xnew_l)**2 + \
                             (np.roll(ynew_l, -1) - ynew_l)**2
                             )
        diff_vec_l = np.concatenate([[0], diff_vec_l[0:point_num - 1]])
        cum_diff_ints = (np.cumsum(diff_vec_l) / ospacing).astype(int)

        locs = cum_diff_ints - np.roll(cum_diff_ints, 1)
        locs[0] = 1

        xnew = xnew_l[locs == 1]
        ynew = ynew_l[locs == 1]

        diff_vec = np.sqrt((np.roll(xnew, -1) - xnew)**2 + \
                           (np.roll(ynew, -1) - ynew)**2)
        max_diff = np.abs((diff_vec[:-1] - 1)).max()
        up_sample += 1


    return np.array((xnew, ynew))


def calc_norms(spline_vec: np.ndarray):
    """
    Calculation of Unit Normal Vectors.
    """
    xsplines, ysplines = spline_vec
    num_point = xsplines.shape[0]

    # initialise vector to store differences
    tans = np.zeros((2, num_point), dtype=np.float32)
    norms = np.zeros((2, num_point), dtype=np.float32)

    # use forward and backward differences on first and last elements where
    # centred difference is ill defined
    tans[:, 0] = (xsplines[1] - xsplines[0], ysplines[1] - ysplines[0])
    tans[:, -1] = (xsplines[-1] - xsplines[-2], ysplines[-1] - ysplines[-2])

    # array manipulation for centred finite differences
    tans[:, 1:-1] = (xsplines[2:] - xsplines[0:-2], ysplines[2:] - ysplines[0:-2])

    # create normals by finding gradients and creating vectors
    diff_arr = tans[1, :] / tans[0, :]

    # this loop creates the normals
    for i in range(num_point):
        if (diff_arr[i] == 0): 
            norms[:, i] = [[0], [1]]
        else:
            norms[:, i] = (1 / np.sqrt(1 + (-1 / diff_arr[i])**2)) * np.array((1, -1/diff_arr[i]))

        # uses the crossproduct to ensure correct orientation of normals
        if (tans[0, i] * norms[1, i]) < (tans[1, i] * norms[0, i]):
            norms[:, i] = -norms[:, i]

    return np.vstack((spline_vec, diff_arr, norms))


def spline_slit(data: np.ndarray, gpoints: np.ndarray, length: int = 10,
                spacing=1,
                diff_tol: float = 1e-2, plot_slit: bool = False):
    """
    Calculates and retrives data along a s specified guide.

    Using a pre-defined set of guide points, the routine calculates a spline
    fit to the guide points. Along the guide spline, the normal vectors are
    determined in order to extract slits perpendicular to the spline with
    of length 2*length+1 pixels.

    Parameters
    ----------
    data - data array in form nt, ny, nx

    gpoints - guide points to define spline (from PointPicker)

    length - half length of slit

    spacing - Euclidean distance between spline points (approximate
          if oversample not used)

    diff_tol - accuracy of spacing. Setting this value smaller leads to longer
               runs times.

    """
    spline_vec = get_spline_points(gpoints.T, spacing=spacing,
                                   diff_tol=diff_tol)

    norm_arr = calc_norms(spline_vec)

    nt, ny, nx = data.shape
    num_points = spline_vec.shape[1]

    # initialises array
    interp_arr = np.zeros((num_points, 2 * length + 1, nt))

    print('Pre-filtering data')
    spline_coeffs = np.zeros((nt, ny, nx))
    for i in np.arange(nt):
        spline_coeffs[i, :, :] = spline_filter(data[i, :, :])

    if plot_slit:
        plt.imshow(data[0], origin='lower')
        plt.plot(spline_vec[0], spline_vec[1], label='Spline')
        plt.plot(gpoints[0], gpoints[1], '+', color='black', label='Guide Points')

    vals = np.arange(2 * length + 1) / (2 * length)
    for i in range(num_points):
        x = (norm_arr[0, i] - length * norm_arr[3, i]) + 2 * length * norm_arr[3, i] * vals
        y = (norm_arr[1, i] - length * norm_arr[4, i]) + 2 * length * norm_arr[4, i] * vals
        for j in range(nt):
            interp_arr[i, :, j] = map_coordinates(spline_coeffs[j],
                                                  np.vstack((y, x)),
                                                  prefilter=False, cval=0.0)
            if plot_slit:
                plt.plot(x, y, 'r')

    if plot_slit:
        plt.plot(x, y, 'r', label='Slits', alpha=0.5)
        plt.legend()

    return interp_arr, (spline_vec, norm_arr)
