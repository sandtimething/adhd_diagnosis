# run pyinstaller to build runAnalysis, and copy modelConfig file to the build destination
# please change path to modelConfig file to the actual path
			 
pyinstaller runAnalysis.py -y --clean --hidden-import pymysql
cp ../config/modelConfig_0904 dist/runAnalysis/modelConfig