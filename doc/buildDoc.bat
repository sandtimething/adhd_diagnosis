:: delete old rst files except index.rst

copy .\source\index.rst .\source\index.cp
del .\source\*.rst
move .\source\index.cp .\source\index.rst

sphinx-apidoc.exe -o .\source ..\py_scripts\

make.bat html