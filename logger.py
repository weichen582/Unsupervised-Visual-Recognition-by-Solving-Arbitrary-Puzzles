import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D',
                 backCount=3, fmt='%(asctime)s- %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = logging.FileHandler(
            filename=filename, mode='a', encoding="utf-8", delay=False)
        th.setFormatter(format_str)

        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('all.log', level='debug')

    log.logger.debug('A')
    log.logger.info('B')
    log.logger.warning('C')
    log.logger.error('D')
    log.logger.critical('E')
