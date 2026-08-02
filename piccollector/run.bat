@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Poly Haven 1024x1024 JPG downloader
echo  双击运行后请保持窗口开启，下载完自动结束
echo ============================================
echo.

py download_polyhaven.py --out images

echo.
if errorlevel 1 (
    echo.
    echo 出错了！请把上方错误信息截图发出来。
) else (
    echo 下载完成！图片保存在 images 文件夹。
)
echo.
pause
