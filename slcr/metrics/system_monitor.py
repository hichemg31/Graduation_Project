import psutil
import os
import time

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def snapshot(self):
        cpu_time = self.process.cpu_times()
        memory = self.process.memory_info()

        return {
            "cpu_user": cpu_time.user,
            "cpu_system": cpu_time.system,
            "rss_mb": memory.rss / (1024 * 1024)
        }
