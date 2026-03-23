from __future__ import annotations

import multiprocessing
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore as TSemaphore
from typing import Any, Callable, Optional


class MulThreading:
    def __init__(self, max_workers: int = 5, print_func: Optional[Callable[[str], None]] = None):
        self.print = print_func or print
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = TSemaphore(max_workers * 3)

    def add_task(self, task: Callable, args, cb: Optional[Callable[[Any], None]]):
        self.semaphore.acquire()

        def handle_result(future):
            self.semaphore.release()
            try:
                result = future.result()
                if cb is not None:
                    cb(result)
            except Exception:
                self.print(f"Error: {traceback.format_exc()}")

        future = self.executor.submit(task, *args)
        future.add_done_callback(handle_result)

    def shutdown(self, force: bool = False):
        self.executor.shutdown(wait=not force, cancel_futures=force)


class MultiProcessor:
    def __init__(
        self,
        name: str = "",
        num_processes: int = int(multiprocessing.cpu_count()),
        print_func: Callable[[str], None] = print,
    ):
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.semaphore = multiprocessing.Manager().Semaphore(num_processes * 2)
        self.name = name
        self.print = print_func

    def add_task(self, task: Callable, args, cb: Optional[Callable[[Any], None]]):
        self.semaphore.acquire()
        self.pool.apply_async(
            self._task_wrap,
            (task, self.print, *args),
            callback=self._callback(cb),
            error_callback=self._error_callback(),
        )

    def _error_callback(self):
        def error_callback(exc):
            self.semaphore.release()
            self.print(f"{self.name}:{exc}")

        return error_callback

    @staticmethod
    def _task_wrap(task: Callable, print_func: Callable[[str], None], *args, **kwargs):
        try:
            return task(*args, **kwargs)
        except Exception:
            print_func(f"pid {os.getpid()}, {traceback.format_exc()}")
            return None

    def _callback(self, original_callback: Optional[Callable[[Any], None]]):
        def callback(result):
            self.semaphore.release()
            if original_callback is not None:
                original_callback(result)

        return callback

    def shutdown(self, force: bool = False):
        if force:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()
