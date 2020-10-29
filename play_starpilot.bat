REM This script open anaconda prompt and runs starpilot (Remember to change anaconda path)
REM https://stackoverflow.com/questions/46305569/how-to-make-batch-files-run-in-anaconda-prompt

call call C:\Users\Tubsp\anaconda3\Scripts\activate.bat C:\Users\Tubsp\anaconda3

python -m procgen.interactive --env-name starpilot

timeout 200