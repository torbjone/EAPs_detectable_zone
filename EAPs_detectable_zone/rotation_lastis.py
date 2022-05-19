import numpy as np

def find_major_axes(cell):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Find the principal geometrical components of the neuron currently loaded
    with Neuron. Uses :class:`sklearn.decomposition.PCA`.
    If used with :class:`LFPy.Cell`, the parameter **pt3d** must be set to True.
    :returns:
        Matrix (3 x 3) where each row is a principal component.
    :rtype: :class:`~numpy.ndarray`
    Example:
        .. code-block:: python
            # Find the principal component axes and rotate cell.
            axes = LFPy_util.data_extraction.findMajorAxes()
            LFPy_util.rotation.alignCellToAxes(cell,axes[0],axes[1])
    """
    from sklearn.decomposition import PCA
    points = np.array([cell.x.mean(axis=1), cell.y.mean(axis=1), cell.z.mean(axis=1)])
    pca = PCA(n_components=3)
    pca.fit(points[:3].T)
    return pca.components_

def alignCellToAxes(cell, y_axis, x_axis=None):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Rotates the cell such that **y_axis** is paralell to the global y-axis and
    **x_axis** will be aligned to the global x-axis as well as possible.
    **y_axis** and **x_axis** should be orthogonal, but need not be.
    :param `~LFPy.Cell` cell:
        Initialized Cell object to rotate.
    :param `~numpy.ndarray` y_axis:
        Vector to be aligned to the global y-axis.
    :param `~numpy.ndarray` x_axis:
        Vector to be aligned to the global x-axis.
    Example:
        .. code-block:: python
            # Find the principal component axes and rotate cell.
            axes = LFPy_util.data_extraction.findMajorAxes()
            LFPy_util.rotation.alignCellToAxes(cell,axes[0],axes[1])
    """
    y_axis = np.asarray(y_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = y_axis[0]
    dy = y_axis[1]
    dz = y_axis[2]

    x_angle = -np.arctan2(dz, dy)
    z_angle = np.arctan2(dx, np.sqrt(dy * dy + dz * dz))

    cell.set_rotation(x_angle, None, z_angle)
    if x_axis is None:
        return

    x_axis = np.asarray(x_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    Rx = rotation_matrix([1, 0, 0], x_angle)
    Rz = rotation_matrix([0, 0, 1], z_angle)

    x_axis = np.dot(x_axis, Rx)
    x_axis = np.dot(x_axis, Rz)

    dx = x_axis[0]
    dz = x_axis[2]

    y_angle = np.arctan2(dz, dx)
    cell.set_rotation(None, y_angle, None)

    if np.abs(np.min(cell.z)) > np.abs(np.max(cell.z)):
        cell.set_rotation(x=np.pi)


def rotation_matrix(axis, theta):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Uses the Euler-rodrigues formula
    """
    theta = -theta
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
