import colorlog
import logging
import sys

import re

def gen_logger(name):
    """ A generalized logger for analysis

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logger : the logger
    """

    logger = logging.getLogger(re.sub(r'^dynosarc\.', '', name))

    # create file handler and set level to debug
    # fh = logging.FileHandler('log.log')
    # fh.setLevel(logging.DEBUG)

    # create console handler and set level higher to INFO
    # ch = logging.StreamHandler(stream=sys.stdout)
    ch = colorlog.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = colorlog.ColoredFormatter(
        fmt="%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(yellow)s%(funcName)s:%(lineno)s%(reset)s | %(process)d >>> %(log_color)s%(message)s%(reset)s",
        datefmt='%m/%d/%Y %I:%M:%S  %p',
    )
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt='[%(asctime)s] %(name)-12s %(funcName)s %(lineno)d %(levelname)-8s %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S  %p')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    ch.setFormatter(fmt)

    # add the handlers to the logger
    # logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def set_verbose(logger, verbose="INFO"):  # "ERROR"):
    """ Set up the verbose level of dynamic console handler
                                                                                
    Parameters  
    ----------                                                                  
    verbose: {"DEBUG","INFO","WARNING","ERROR"}
        Verbose level. (Default = "INFO")
            - "DEBUG": show all output logs. 
            - "INFO": show only iteration process log.
            - "WARNING": show warnings and errors.
            - "ERROR": only show log if error happened. 
    """
    if verbose == "DEBUG":
        # ch.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose == "INFO":
        # ch.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    elif verbose == "WARNING":
        # ch.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
    elif verbose == "ERROR":
        # ch.setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)
    else:
        # print("Unrecognized verbose level, options: ['DEBUG','INFO','WARNING','ERROR'], use 'INFO' instead")
        logger.setLevel(logging.INFO)

