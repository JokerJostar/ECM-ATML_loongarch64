# loongarch64-toolchain.cmake

# 设置编译器路径
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

# 设置工具链前缀
set(CMAKE_C_COMPILER /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-g++)

# 设置其他工具
set(CMAKE_AR /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-ar)
set(CMAKE_AS /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-as)
set(CMAKE_NM /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-nm)
set(CMAKE_RANLIB /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-ranlib)
set(CMAKE_STRIP /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/loongarch64-linux-gnu-strip)

# 设置查找路径
set(CMAKE_FIND_ROOT_PATH /home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1)

# 设置查找策略
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

