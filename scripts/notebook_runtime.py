"""Запуск ноутбука тем же Python, что выполняет сборку"""
import os
import sys
from jupyter_client import AsyncKernelManager
from audit_source import ROOT

os.environ.setdefault('IPYTHONDIR',str(ROOT/'data/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'data/jupyter-runtime'))


class ProjectKernelManager(AsyncKernelManager):
    async def start_kernel(self, **kwargs):
        # Не допускаем незаметного переключения на системный Python
        self.kernel_spec.argv[0] = sys.executable
        return await super().start_kernel(**kwargs)
