import numpy as np

def compress_image(channel_data_matrix, singular_values_limit):
    """
    Helper to use SVD to compress an image to smaller size based on
    `singular_values_limit`
    """
    U, s, V = np.linalg.svd(channel_data_matrix) 
    return np.array(np.matrix(U[:, :singular_values_limit]) * 
                    np.diag(s[:singular_values_limit]) * 
                    np.matrix(V[:singular_values_limit, :]))