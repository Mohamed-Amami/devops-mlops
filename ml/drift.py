from scipy.stats import ks_2samp

def detect_drift(old_data, new_data):
    stat, p_value = ks_2samp(old_data, new_data)
    return p_value < 0.05
