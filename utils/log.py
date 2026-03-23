from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional


def init_logger(
    logger_obj: logging.Logger,
    level: int = logging.DEBUG,
    level_file: int = logging.DEBUG,
    consol_level: int = logging.DEBUG,
    logfile: Optional[str] = None,
):
    logger_obj.setLevel(level)

    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")

    if not logfile:
        logfile_path = Path(f"./logs/{time_string}.log")
        logfile_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        logfile_path = Path(logfile)
        logfile_path.parent.mkdir(exist_ok=True, parents=True)

    if logger_obj.handlers:
        logger_obj.handlers.clear()

    file_handler = logging.FileHandler(logfile_path, mode="a")
    file_handler.setLevel(level_file)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(consol_level)

    formatter = logging.Formatter("[%(asctime)s - %(filename)s, line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(console_handler)


logger = logging.getLogger("CC-HARD")
