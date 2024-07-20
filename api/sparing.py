import logging
from datetime import datetime

from scipy import stats


class DataDriftDetector:
    def __init__(self, reference_data, model, threshold=0.05):
        self.reference_data = reference_data
        self.model = model
        self.threshold = threshold
        self.feature_names = model.feature_names_in_

        # Set up logging
        logging.basicConfig(filename='data_drift.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    def detect_drift(self, new_data):
        drift_results = {}
        missing_features = []

        for feature in self.feature_names:
            if feature not in new_data.columns:
                missing_features.append(feature)
                continue

            if self.reference_data[feature].dtype in ['int64', 'float64']:
                ks_statistic, p_value = stats.ks_2samp(self.reference_data[feature], new_data[feature])
                drift_detected = p_value < self.threshold

                drift_results[feature] = {
                    'drift_detected': drift_detected,
                    'p_value': p_value,
                    'ks_statistic': ks_statistic
                }

        self._log_drift_results(drift_results, missing_features)
        return drift_results, missing_features

    def _log_drift_results(self, drift_results, missing_features):
        drifted_features = [feature for feature, result in drift_results.items() if result['drift_detected']]

        if drifted_features or missing_features:
            log_message = f"Data drift detected at {datetime.now()}:\n"
            if drifted_features:
                log_message += f"Drifted features: {', '.join(drifted_features)}\n"
            if missing_features:
                log_message += f"Missing features: {', '.join(missing_features)}\n"
            logging.info(log_message)


def handle_missing_features(new_data, feature_names):
    for feature in feature_names:
        if feature not in new_data.columns:
            # Add the missing feature with a default value (e.g., mean or median from training data)
            # For this example, we'll use 0 as a placeholder. In practice, you should use a more informed approach.
            new_data[feature] = 0
    return new_data[feature_names]
