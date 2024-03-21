1.  Run msys2_shell.cmd from C:\ghcup\msys64
2.  pacman -Sy 
3.  pacman -S mingw-w64-x86_64-gcc-fortran
4.  pacman -S mingw-w64-x86_64-openblas
5.  pacman -S mingw-w64-x86_64-gsl
6.  pacman -S mingw-w64-x86_64-glpk
7.  Run cmd
8.  cabal update
9.  cabal install hmatrix --flag=openblas --extra-lib-dirs=C:\ghcup\msys64\mingw64\lib --extra-include-dirs=C:\ghcup\msys64\mingw64\include
10. cabal install --lib hmatrix --flag=openblas --extra-lib-dirs=C:\ghcup\msys64\mingw64\lib --extra-include-dirs=C:\ghcup\msys64\mingw64\include
11. Add C:\ghcup\msys64\mingw64\bin to PATH
