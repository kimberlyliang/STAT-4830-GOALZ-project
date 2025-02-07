import os

class ModelConfig:
    MODEL = "moirai"  # we can choose from {'moirai', 'moirai-moe'} not sure which one is better
    SIZE = "small"    # also have {'small', 'base', 'large'}
    PREDICTION_LENGTH = 20  # prediction window
    CONTEXT_LENGTH = 200   # context window
    PATCH_SIZE = "auto"   # patch size for attention -> auto does a bunch of different sizes which is what I think we want
    BATCH_SIZE = 32
    NUM_SAMPLES = 100
    
class DataConfig:
    BASE_PATH = "/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/Anphy Dataset"
    DETAILS_CSV = BASE_PATH + "/Details information for healthy subjects.csv"
    ARTIFACT_PATH = BASE_PATH + "/Artifact matrix"

    SUBJECTS = [f"EPCTL{str(i).zfill(2)}" for i in range(1, 30)]

    @staticmethod
    def get_subject_paths(subject_id):
        return {
            'edf': f"{DataConfig.BASE_PATH}/{subject_id}/{subject_id}.edf",
            'txt': f"{DataConfig.BASE_PATH}/{subject_id}/{subject_id}.txt",
            'artifact': f"{DataConfig.ARTIFACT_PATH}/{subject_id}_artndxn.mat"
        }
    
    @staticmethod
    def ensure_subject_results_dir(subject_id):
        """Create subject-specific results directory if it doesn't exist"""
        subject_results_dir = DataConfig.get_subject_paths(subject_id)['results']
        if not os.path.exists(subject_results_dir):
            os.makedirs(subject_results_dir)
        return subject_results_dir
    
    TRAIN_SPLIT = 0.8