# params.py


# Path to the file that contains the learning rates to avoid
AVOID_FILE_PATH = './avoid.txt'

# Directory where the training data is stored
DATA_DIR = './train_data'

# Path where the trained model will be saved
MODEL_SAVE_PATH = 'temp/saved_model/saved'

# The input size of the model, which is the length of each ECG segment
INPUT_SIZE = 1250

# The number of epochs for training the model
NUM_EPOCHS = 40

# The size of the batches used in training and validation
BATCH_SIZE = 100

# The minimum, maximum, and step size for generating learning rates
LR_MIN = 0.0001  # Minimum learning rate to try
LR_MAX = 0.0050   # Maximum learning rate to try
STEP = 0.00005   # Step size for incrementing the learning rate

# The number of worker threads to use for loading data
NUM_WORKERS = 8  # params on v100
# NUM_WORKERS = 8 # params on pc

# The number of batches that should be prefetched by each worker
PREFETCH_FACTOR = 4   # params on v100
# PREFETCH_FACTOR = 2 # params on pc

# Deciding when to write the current learning_rate to avoid.txt
AVOID_PARAM = 0.9

