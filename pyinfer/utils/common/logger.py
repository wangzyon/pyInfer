#!/usr/bin/env python
import logging
from logging.handlers import RotatingFileHandler
import colorlog

__all__ = ["Logger"]


class Logger():
    """日志"""
    instance = None

    def __new__(cls, filename=None, level="INFO"):
        """
        日志设计为单例模式

        注：
        getLogger对于相同filename返回同一对象，需要保证addHandler仅对同一logger对象进行一次，否则日志将出现重复打印；单例可确保上述需求；
        
        """

        level_dict = {"INFO": logging.INFO, "DEBUG": logging.DEBUG, "FATAL": logging.FATAL, "ERROR": logging.ERROR}

        if cls.instance is None:
            cls.instance = super().__new__(cls)    # 未经过初始化的实例对象
            cls.instance._logger = cls.getLogger(filename, level_dict.get(level))
        return cls.instance

    @classmethod
    def getLogger(cls, filename, level):
        logger = logging.getLogger(filename)

        log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'FATAL': 'red',
        }

        formatter = colorlog.ColoredFormatter(
            '%(log_color)s [%(levelname)s] %(asctime)s  %(filename)s[line:%(lineno)d] : %(message)s',
            log_colors=log_colors_config)

        # 设置日志级别
        logger.setLevel(logging.INFO)
        # 往屏幕上输出
        console_handler = logging.StreamHandler()
        # 设置屏幕上显示的格式
        console_handler.setFormatter(formatter)
        # 把对象加到logger里
        logger.addHandler(console_handler)

        # 输出到文件
        if filename is not None:
            file_handler = RotatingFileHandler(filename=filename, mode='a', maxBytes=1 * 1024 * 1024, encoding='utf8')
            file_formatter = logging.Formatter(
                '[%(levelname)s] %(asctime)s  %(filename)s[line:%(lineno)d]: %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        return logger

    def warning(self, msg):
        self._logger.warning(msg)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)

    def fatal(self, msg):
        self._logger.fatal(msg)
        raise