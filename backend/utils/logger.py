"""
SEED AI — Structured JSON Logger (Production v2)
Outputs structured JSON to stdout + persistent file logs with rotation.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    JSON structured logger for Observability.
    Captures traces for Decision, Agent, Tool, Memory, and Error.
    """

    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # Vercel has a read-only filesystem, so we skip file logs
    if not os.environ.get("VERCEL"):
        os.makedirs("logs", exist_ok=True)

        file_handler = RotatingFileHandler(
            "logs/seed_ai.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        error_handler = RotatingFileHandler(
            "logs/errors.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=3,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)

        correlation_handler = RotatingFileHandler(
            "logs/execution_traces.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        correlation_handler.setLevel(logging.INFO)
        correlation_handler.setFormatter(JSONFormatter())
        logger.addHandler(correlation_handler)

    return logger
