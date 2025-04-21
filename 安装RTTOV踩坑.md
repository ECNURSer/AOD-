报错：configure: error: Can't find or link to the z library. Turn off netCDF-4 and DAP clients with --disable-hdf5 --disable-dap, or see config.log for errors.

解决方案

```sh
sudo CPPFLAGS="-I${H5DIR}/include -I/usr/local/zlib/include" LDFLAGS="-L${H5DIR}/lib -L/usr/local/zlib/lib" ./configure --prefix=${NCDIR} --disable-libxml2
```

报错：Fatal Error: Can't open module file 'hdf5.mod' for reading at (1): No such file or directory
make[1]: *** [../.././/mod/rttov_hdf_mod.mod] Error 1
make[1]: Leaving directory `/usr/local/rttov132/src/hdf'
make: *** [hdf/lib] Error 2

```sh
$ ./configure  --with-zlib=${ZDIR} --prefix=${H5DIR} FC=gfortran CC=gcc --enable-fortran
```

报错：checking size of off_t... configure: error: in `/home/hd/rttov_needed/netcdf-fortran-4.6.1':
configure: error: cannot compute sizeof (off_t)

```sh
 sudo CPPFLAGS=-I${NCDIR}/include LDFLAGS=-L${NCDIR}/lib ./configure --prefix=${NCDIR} LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} FC=gfortran

```

报错： /usr/local/rttov132/rttov_test/../bin/rttov_test.exe: error while loading shared
		    | libraries: libnetcdff.so.7: cannot open shared object file: No such file or direc

这个错误表明在运行 `rttov_test.exe` 时，找不到 `libnetcdff.so.7` 共享库。这个问题通常是由于库路径未正确设置导致的。以下是一些解决方法：

### 解决方法

1. **找到 `libnetcdff.so.7` 文件**：
    ```sh
    find / -name libnetcdff.so.7
    ```

    这将帮助你找到 `libnetcdff.so.7` 的实际位置。

2. **设置 `LD_LIBRARY_PATH`**：
    假设 `libnetcdff.so.7` 位于 `/usr/local/netcdf/lib` 目录中，你可以这样设置环境变量：

    ```sh
    export LD_LIBRARY_PATH=/usr/local/netcdf/lib:$LD_LIBRARY_PATH
    ```

    为了使这个设置在每次登录时都生效，可以将上述命令添加到你的 `~/.bashrc` 或 `~/.bash_profile` 文件中。

3. **验证设置**：
    运行以下命令以验证 `LD_LIBRARY_PATH` 是否设置正确：

    ```sh
    echo $LD_LIBRARY_PATH
    ```
