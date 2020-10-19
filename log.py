# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:20:51 2020

@author: 徐钦华
"""
import sys
sys.path.append(r'D:\model_train\logging')
import logging
import logging.handlers
import datetime
ALL_LOG=r'D:\model_train\logging\ALL_LOG.txt'
ERR_LOG=r'D:\model_train\logging\ERR_LOG.txt'
class logger():
    def __init__(self):
        self.logger = logging.getLogger('job_manage')
        self.logger.setLevel(logging.DEBUG)
    
        rf_handler = logging.handlers.TimedRotatingFileHandler(ALL_LOG, when='midnight', interval=1, backupCount=7,
                                                               atTime=datetime.time(0, 0, 0, 0))
        rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
        f_handler = logging.FileHandler(ERR_LOG)
        ch = logging.StreamHandler()
    
        f_handler.setLevel(eval("logging.%s" % 'ERROR'))
        f_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    
        ch.setLevel(eval("logging.%s" % 'INFO'))
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    
        self.logger.addHandler(rf_handler)
        self.logger.addHandler(f_handler)
        self.logger.addHandler(ch)
        

logger = logger().logger


