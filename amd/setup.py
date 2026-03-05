from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='FlashAllToAll',
    ext_modules=[
        CppExtension('FlashAllToAll', ['flash.cpp',
            "./src/fast_alltoall/alltoall_algorithm.cc",
            "./src/fast_alltoall/alltoall_global_scheduler.cc",
            "./src/fast_alltoall/alltoall_local_scheduler.cc",
            "./src/fast_alltoall/alltoall_matrix.cc",
            "./src/fast_alltoall/include/fast_alltoall/alltoall_algorithm.h",
            "./src/fast_alltoall/include/fast_alltoall/alltoall_global_scheduler.h",
            "./src/fast_alltoall/include/fast_alltoall/alltoall_local_scheduler.h",
            "./src/fast_alltoall/include/fast_alltoall/alltoall_matrix.h",
            "./src/fast_alltoall/include/fast_alltoall/alltoall_define.h"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
