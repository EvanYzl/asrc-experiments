class MbertParams(object):
    CUDA_NUM = 0
    """ used GPU num """

    MODEL_INPUT_DIM  = 768
    MODEL_OUTPUT_DIM = 300
    """ dimension of basic bert unit output embedding """
    RANDOM_DIVIDE_ILL = False
    """ if True: get train/test_ILLs by random divide all entity ILLs,
    else: get train/test ILLs from file."""
    TRAIN_ILL_RATE = 0.3
    """ (only work when RANDOM_DIVIDE_ILL == True) training data rate. 
    Example: train ILL number: 15000 * 0.3 = 4500."""

    SEED_NUM = 11037

    EPOCH_NUM = 10
    """ training epoch num """

    NEAREST_SAMPLE_NUM = 128
    CANDIDATE_GENERATOR_BATCH_SIZE = 128

    TOPK = 50
    NEG_NUM = 5
    """ negative sample num """
    MARGIN = 3
    LEARNING_RATE = 2e-5
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 128

    DES_LIMIT_LENGTH = 32
    """ max length of description/name. """
