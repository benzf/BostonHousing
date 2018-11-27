import numpy as np


class PerformanceMetrics:
    """Defines methods to evaluate the model
    Parameters
    ----------
    y_actual : array-like, shape = [n_samples]
            Observed values from the training samples
    y_predicted : array-like, shape = [n_samples]
            Predicted values from the model
    """ 

    def compute_rmse(self,y_actual, y_predicted):
        """Compute the root mean squared error
        Returns
        ------
        rmse : root mean squared error
        """
        return np.sqrt(self.sum_of_square_of_residuals(y_actual, y_predicted))

    def compute_r2_score(self, y_actual, y_predicted):
        """Compute the r-squared score
            Returns
            ------
            r2_score : r-squared score
            """
        # sum of square of residuals
        ssr = self.sum_of_square_of_residuals(y_actual, y_predicted)

        # total sum of errors
        sst = np.sum((y_actual - np.mean(y_actual)) ** 2)

        return 1 - (ssr / sst)

    def sum_of_square_of_residuals(self, y_actual, y_predicted):
        return np.sum((y_actual - y_predicted) ** 2)
