@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
"C:\Program Files\Python38\python.exe" -m pip install -r requirements.txt
"C:\Program Files\Python38\python.exe" -m streamlit run app.py 