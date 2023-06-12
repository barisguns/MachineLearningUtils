from sklearn.base import BaseEstimator, RegressorMixin
from scipy.spatial.transform import Rotation


class KabschRotationRegressor(BaseEstimator, RegressorMixin):
    """
    This is a custom sklearn regressor which uses Kabsch algorithm to learn a rotation matrix, from a given matrix
    to another target matrix, then uses this rotation to regress other matrices in the direction of the target matrix.
    Uses scipy implementation of Kabsch algorithm
    """
    def __init__(self):
        """
        Construct a rotation regressor which learns a rotation using Kabsch algorithm.
        """
        self.rotation = None
        return

    def fit(self, mat_to_be_rotated, rot_direction_mat=None):
        """
        Learn the rotation matrix using scipy implementation of Kabsch algorithm.
        :param mat_to_be_rotated:
        :param rot_direction_mat:
        :return:
        """
        self.rotation = Rotation.align_vectors(rot_direction_mat, mat_to_be_rotated)
        return

    def predict(self, mat_to_be_rotated):
        """
        Rotates the given matrix using the learned rotation matrix in the direction of the target matrix.
        :param mat_to_be_rotated:
        :return:
        """
        rotated_mat = self.rotation[0].apply(mat_to_be_rotated)
        return rotated_mat


