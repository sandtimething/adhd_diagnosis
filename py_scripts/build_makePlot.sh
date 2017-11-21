# install gtk3-runtime first
# https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
			 
pyinstaller makePlot.py -y --clean --hidden-import pymysql --hidden-import pygal --hidden-import loadData --hidden-import cairocffi --hidden-import mathHelper -p 'D:\mygit\rnd_analysis\utilities'
mkdir -p ./dist/makePlot/pygal/css/ && cp C:/Users/Wang/Anaconda3/Lib/site-packages/pygal/css/* "$_"
# cp ./zlib1.dll ./dist/makePlot/zlib1.dll
# cp ./libcairo-2.dll ./dist/makePlot/libcairo-2.dll