#!/usr/bin/env python
# encoding: utf-8

"""
Use sklearn based API model to local run and tuning.
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import logging
import logging.config


def start_logging():
    # 加载前面的标准配置
    from ProjectCodes.logging_config import ConfigLogginfDict
    logging.config.dictConfig(ConfigLogginfDict(__file__).LOGGING)
    # 获取loggers其中的一个日志管理器
    logger = logging.getLogger("default")
    logger.info('\n\n#################\n~~~~~~Start~~~~~~\n#################')
    print(type(logger))
    return logger
if 'Logger' not in dir():
    Logger = start_logging()


class LocalRegressor(BaseEstimator, RegressorMixin):
    """ An sklearn-API regressor.
    Model 1: Embedding GRU ---- Embedding(text or cat) -> Concat[GRU(words) or Flatten(cat_vector)] ->  Dense -> Output
    Parameters
    ----------
    demo_param : All tuning parameters should be set in __init__()
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """

    def __init__(self, name_emb_dim=20, item_desc_emb_dim=60, cat_name_emb_dim=20, brand_emb_dim=10, cat_main_emb_dim=10,
                 cat_sub_emb_dim=10, cat_sub2_emb_dim=10, item_cond_id_emb_dim=5, GRU_layers_out_dim=(16, 8, 8),
                 drop_out_layers=(0.25, 0.1), dense_layers_dim=(128, 64)):
        self.name_emb_dim = name_emb_dim
        self.item_desc_emb_dim = item_desc_emb_dim
        self.cat_name_emb_dim = cat_name_emb_dim
        self.brand_emb_dim = brand_emb_dim
        self.cat_main_emb_dim = cat_main_emb_dim
        self.cat_sub_emb_dim = cat_sub_emb_dim
        self.cat_sub2_emb_dim = cat_sub2_emb_dim
        self.item_cond_id_emb_dim = item_cond_id_emb_dim
        self.GRU_layers_out_dim = GRU_layers_out_dim
        self.drop_out_layers = drop_out_layers
        self.dense_layers_dim = dense_layers_dim
        self.emb_GRU_model = get_GRU_model()


if __name__ == "__main__":
    # Get Data include some pre-process.
    sample_df, last_valida_df, test_df = read_file_preproc()